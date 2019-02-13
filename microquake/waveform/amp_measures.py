import numpy as np

from microquake.core.util.tools import copy_picks_to_dict
from microquake.core.stream import Stream
from microquake.waveform.pick import calculate_snr
from microquake.core.data.inventory import get_sensor_type_from_trace

""" This module's docstring summary line.
    This is a multi-line docstring. Paragraphs are separated with blank lines.

    Lines conform to 79-column limit.

    Module and packages names should be short, lower_case_with_underscores.

         1         2         3         4         5         6         7
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""
'''
def measure_pick_amps(st, cat, phase_list=['P'], debug=False, \
                      min_pulse_width=.0016, min_pulse_snr=5, \
                      use_stats_dict=False):
'''

def measure_pick_amps(st, cat, phase_list=['P'], **kwargs):

    """
    For each tr in st, for each pick phase (P,S) in picks:
        Measures the polarity and zero crossing from velocity trace
        Measures the pulse width and area from displacement trace
        Measures the arrival peak amps and times for acc, vel, disp 

    All measurements are added to the arrival extras dict

    :param st: velocity traces
    :type st: either obspy.core.Stream or microquake.core.Stream
    :param cat: obspy catalog
    :type list: list of obspy events
    """

    fname = 'measure_pick_amps'

    measure_velocity_pulse(st, cat, phase_list=phase_list, **kwargs)
                           #min_pulse_width=min_pulse_width, min_pulse_snr=min_pulse_snr, \
                           #use_stats_dict=use_stats_dict)
    measure_displacement_pulse(st, cat, phase_list=phase_list, debug=kwargs['debug'])
    #measure_displacement_pulse(st, cat, phase_list=phase_list, debug=debug, use_stats_dict=use_stats_dict)

    return


def measure_velocity_pulse(st, cat, phase_list=['P'], debug=False, \
                           min_pulse_width=.02, min_pulse_snr=7, \
                           use_stats_dict=False):
    """
    locate velocity pulse (zero crossings) near pick and measure peak amp, polarity, etc on it

    velocity pulse measurements are stored in arrival extras dict

    :param st: velocity traces
    :type st: either obspy.core.Stream or microquake.core.Stream
    :param cat: obspy catalog
    :type list: list of obspy events
    """

    fname = 'measure_velocity_pulse'

  # TODO: Move these to toml settings
  # Min thresh params for velocity pulses:
    pulse_thresh_P = 9.
    pulse_thresh_S = 6.
    pulse_min_width = .0014


    for event in cat:
        arrivals = event.preferred_origin().arrivals
        picks = [ arr.pick_id.get_referred_object() for arr in arrivals ]
        pick_dict = copy_picks_to_dict(picks)

        for tr in st:

            sta = tr.stats.station

        # TODO: Note that tr channel codes may change (e.g., enz --> P,SV,SH) then this call will fail!
            sensor_type = get_sensor_type_from_trace(tr)

            if sensor_type != "VEL":
                print("%s: tr:%s units:%s NOT VEL --> Skip polarity check" % (fname, tr.get_id(), sensor_type))
                continue

            tr.detrend("demean").detrend("linear")
            data = tr.data.copy()

            for phase in phase_list:

                if debug:
                    print("measure_vel_pulse: sta:%s cha:%s pha:%s" % (sta, tr.stats.channel, phase))

                if sta not in pick_dict or phase not in pick_dict[sta]:
                    print("%s: sta:%s has no [%s] pick" % (fname, sta, phase))
                    continue

                pick = pick_dict[sta][phase]

                arrival = get_arrival_from_pick(arrivals, pick)

                # This should never occur since we got the picks *from* the arrivals[]
                if arrival is None:
                    print("%s: Unable able to locate arrival for sta:%s pha:%s pick" % (fname, sta, phase))
                    continue

                pick_time = pick_dict[sta][phase].time

                ipick = int((pick_time - tr.stats.starttime) * tr.stats.sampling_rate)

                #print("Call find_signal_zeros: sta:%s pick_time=%s" % (sta, pick_time))
                # This setting is for UK
                polarity, vel_zeros = find_signal_zeros(tr, ipick, nzeros_to_find=3, \
                                                        min_pulse_width=min_pulse_width,\
                                                        min_pulse_snr=min_pulse_snr, debug=debug)
                # This setting is for OT
                #polarity, vel_zeros = find_signal_zeros(tr, ipick, nzeros_to_find=3, min_pulse_width=.0016)

                dd = {}
                dd['pick_time']= pick_time

                stats_key = "%s_arrival" % phase
                if use_stats_dict:
                    if stats_key not in tr.stats:
                        tr.stats[stats_key] = {}

                # A good pick will have the first velocity pulse located between i1 and i2
                if vel_zeros is not None:
                    i1 = vel_zeros[0]
                    i2 = vel_zeros[1]
                    t1 = tr.stats.starttime + float(i1 * tr.stats.delta)
                    t2 = tr.stats.starttime + float(i2 * tr.stats.delta)

                    ipeak, peak_vel = get_peak_amp(tr, i1, i2)
                    tpeak = tr.stats.starttime + float(ipeak * tr.stats.delta)

                    noise_npts  = int(.01 * tr.stats.sampling_rate)
                    noise_end   = ipick - int(.005 * tr.stats.sampling_rate)
                    #noise_level = np.std(data[noise_end - noise_npts: noise_end])
                    noise = data[noise_end - noise_npts: noise_end]
                    noise1=np.abs(np.mean(noise))
                    noise2=np.abs(np.median(noise))
                    noise3=np.abs(np.std(noise))
                    noise_level = np.max([noise1, noise2, noise3])

                    pulse_snr = np.abs(peak_vel/noise_level)
                    pulse_width = float((i2-i1)*tr.stats.delta)
                    #print("pulse_snr:%.1f = peak_vel=%g / noise=%g" % (pulse_snr, peak_vel, noise_level))

                    pulse_thresh = pulse_thresh_P

                    #print("sta:%s pha:%s t1=%s t2=%s polarity=%d" % (sta, phase, t1, t2, polarity))
                    #tr.plot()

                    if phase == 'S':
                        pulse_thresh = pulse_thresh_S

                    if pulse_snr < pulse_thresh:
                        print("%s: tr:%s pha:%s t1:%s t2:%s pulse_snr=%.1f < thresh" % \
                            (fname, tr.get_id(), phase, t1, t2, pulse_snr))
                        polarity = 0

                    if pulse_width < pulse_min_width:
                        print("%s: tr:%s pha:%s t1:%s t2:%s pulse_width=%f < .0014" % \
                            (fname, tr.get_id(), phase, t1, t2, pulse_width))
                        polarity = 0


                    if use_stats_dict:
                        dd['polarity'] = polarity
                        dd['peak_vel'] = peak_vel
                        dd['tpeak'] = tpeak
                        dd['t1'] = t1
                        dd['t2'] = t2
                        dd['pulse_snr'] = pulse_snr
                    else:
                        arrival.polarity  = polarity
                        arrival.peak_vel  = peak_vel
                        arrival.tpeak_vel = tpeak
                        arrival.t1 = t1
                        arrival.t2 = t2
                        arrival.pulse_snr = pulse_snr

                else:
                    print("%s: Unable to locate zeros for tr:%s pha:%s" % (fname,tr.get_id(),phase))
                    polarity = 0
                    if use_stats_dict:
                        dd['polarity'] = polarity
                    else:
                        arrival.polarity  = polarity

                if use_stats_dict:
                    tr.stats[stats_key]['velocity_pulse'] = dd

            # Process next phase in phase_list

        # Process tr in st

    # Process next event in cat


    return



def measure_displacement_pulse(st, cat, phase_list=['P'], debug=False, use_stats_dict=False):
    """
    measure displacement pulse (area + width) for each pick on each trace,
        as needed for moment magnitude calculation

    displacement pulse measurements are stored in each arrival extras dict

    :param st: velocity traces
    :type st: either obspy.core.Stream or microquake.core.Stream
    :param cat: obspy catalog
    :type list: list of obspy events
    """

    fname = 'measure_displacement_pulse'

    for event in cat:
        arrivals = event.preferred_origin().arrivals
        picks = [ arr.pick_id.get_referred_object() for arr in arrivals ]
        pick_dict = copy_picks_to_dict(picks)

        for tr in st:

            sta = tr.stats.station

            sensor_type = get_sensor_type_from_trace(tr)

            if sensor_type != "VEL":
                print("%s: tr:%s units:%s NOT VEL --> Skip" % (fname, tr.get_id(), sensor_type))
                continue

            tr_dis = tr.copy().detrend("demean").detrend("linear")
            tr_dis.integrate().detrend("linear")
            tr_dis.stats.channel="%s.dis" % tr.stats.channel

            for phase in phase_list:

                if debug:
                    print("measure_dis_pulse: sta:%s cha:%s pha:%s" % (sta, tr.stats.channel, phase))

                if phase not in pick_dict[sta]:
                    print("%s: sta:%s has no [%s] pick in dict" % (fname, sta, phase))
                    continue

                pick = pick_dict[sta][phase]
                arrival = get_arrival_from_pick(arrivals, pick)
                if arrival is None:
                    print("%s: Unable able to locate arrival for sta:%s pha:%s pick" % (fname, sta, phase))

                dd = {}

                stats_key = "%s_arrival" % phase

                if use_stats_dict:
                    if stats_key in tr.stats and 'velocity_pulse' in tr.stats[stats_key]:
                        #print(tr.stats)
                        polarity = tr.stats[stats_key]['velocity_pulse']['polarity']
                        t1 = tr.stats[stats_key]['velocity_pulse']['t1']
                        t2 = tr.stats[stats_key]['velocity_pulse']['t2']

                    else:
                        print("%s: tr:%s --> NOT found velocity_pulse in stats_key=%s" % (fname, tr.get_id(), stats_key))
                        continue
                else:
                    polarity = arrival.polarity
                    t1 = arrival.t1
                    t2 = arrival.t2

                if t1 is None:
                    print("%s: tr:%s velocity pulse does not seem set --> Skip displacment measure" % \
                          (fname, tr.get_id()))
                    continue

                pick_time = pick_dict[sta][phase].time

                i1 = int((t1 - tr.stats.starttime) * tr.stats.sampling_rate)
                i2 = int((t2 - tr.stats.starttime) * tr.stats.sampling_rate)

                ipick = int((pick_time - tr.stats.starttime) * tr.stats.sampling_rate)

                #print("measure_dis_pulse: sta:%s [%s] t1:%s t2:%s i1:%d i2:%d sr:%f" % \
                      #(sta,phase,t1,t2,i1,i2,tr.stats.sampling_rate))

                #tr.plot()
                #exit()

                #print("measure_dis_pulse: polarity=%d" % polarity)

                if polarity != 0:

                    icross = i2
                    tr_dis.data = tr_dis.data - tr_dis.data[i1]
                    #tr_dis.data = tr_dis.data - tr_dis.data[ipick]

                    dis_polarity = np.sign(tr_dis.data[icross])
                    pulse_width, pulse_area = get_pulse_width_and_area(tr_dis, i1, icross)

                    npulse = int(pulse_width * tr.stats.sampling_rate)

                    max_pulse_duration = .08
                    nmax_len = int(max_pulse_duration * tr.stats.sampling_rate)

                    if pulse_width != 0:

                        # peak_dis = peak of first pulse
                        ipeak,peak_dis = get_peak_amp(tr_dis, ipick, ipick + npulse)
                        # max_dis = max within max_pulse_duration of pick time
                        imax, max_dis  = get_peak_amp(tr_dis, ipick, ipick + nmax_len)

                        tmax_dis  = tr.stats.starttime + float(imax * tr.stats.delta)
                        tpeak_dis = tr.stats.starttime + float(ipeak * tr.stats.delta)
                        tcross_dis = pick_time + pulse_width

                        if use_stats_dict:

                            dd['peak_dis'] = peak_dis
                            dd['max_dis']  = max_dis
                            dd['tpeak_dis'] = tpeak_dis
                            dd['tmax_dis']  = tmax_dis
                            dd['dis_pulse_width'] = pulse_width
                            dd['dis_pulse_area']  = pulse_area

                        else:

                            arrival.peak_dis = peak_dis
                            arrival.max_dis  = max_dis
                            arrival.tpeak_dis = tpeak_dis
                            arrival.tmax_dis  = tmax_dis
                            arrival.dis_pulse_width = pulse_width
                            arrival.dis_pulse_area  = pulse_area


                        if debug:
                            print("[%s] Dis pol=%d tpick=%s" % \
                                (phase, dis_polarity, pick_time))
                            print("              tpeak=%s peak_dis=%12.10g" % (tpeak_dis, peak_dis))
                            print("             tcross=%s" % tcross_dis)
                            print("               tmax=%s max_dis=%12.10g" % (tmax_dis, max_dis))
                            print("    dis pulse width=%.5f" % pulse_width)
                            print("    dis pulse  area=%12.10g" % pulse_area)

                    else:
                        print("Got pulse_width=0 for tr:%s pha:%s" % (tr.get_id(), phase))
                else:
                    print("Got polarity=0 for tr:%s pha:%s" % (tr.get_id(), phase))

                if use_stats_dict:
                    tr.stats[stats_key]['displacement_pulse'] = dd
                    #print(tr.stats)
                    #exit()

            # Process next phase in phase_list

        # Process tr in st

    # Process next event in cat


    return



def find_signal_zeros(tr, istart, max_pulse_duration=.1, nzeros_to_find=3, \
                      second_try=False, min_pulse_width=.00167, min_pulse_snr=5,
                      debug=False):

    fname = 'find_signal_zeros'

    data = tr.data
    sign = np.sign(data)

    counter = 0
    s = 0
    i1 = -9

    if second_try:
        print("%s: This is the second try!" % fname)

    noise_tlen  = .05
    noise_npts  = int(noise_tlen * tr.stats.sampling_rate)
    noise_end   = istart - int(.005 * tr.stats.sampling_rate)
    #noise_level = np.mean(data[noise_end - noise_npts: noise_end])
    noise = data[noise_end - noise_npts: noise_end]
    noise1=np.abs(np.mean(noise))
    noise2=np.abs(np.median(noise))
    noise3=np.abs(np.std(noise))
    noise_level = np.max([noise1, noise2, noise3])
# MTH: OT seems to need noise_level = mean, while
#      UK needs noise_level = max ??
    noise_level = noise1
    #print("noise: mean:%g med:%g std:%g max:%g" % (noise1, noise2, noise3, noise_level))

    #print("noise_start=%d noise_npts=%d noise_end=%d istart=%d" % ((noise_end-noise_npts), noise_npts, noise_end, istart))
    pick_snr = np.abs(data[istart]/noise_level)
    #print("noise_level=%g pick_snr=%f" % (noise_level, pick_snr))

    nmax_look = int(max_pulse_duration * tr.stats.sampling_rate)

    # Just used for debug
    pick_time = tr.stats.starttime + float(istart*tr.stats.delta)

# Stage 0: Take polarity sign (s0) from first data point after istart (=ipick) with 
#          SNR >= thresh * noise_level

    s0 = 0
    i0 = 0
    snr_thresh = 10.
    for i in range(istart, istart + nmax_look):
        #print("i:%4d data[i]:%g snr:%f" % (i,data[i],np.abs(data[i]/noise_level)))
        if np.abs(data[i]) >= snr_thresh * np.abs(noise_level):
            #print("  data[i]=%g >= 10 x noise_level --> Set sign s0=%d" % (data[i], sign[i]))
            s0 = sign[i]
            i0 = i
            break

# Stage 1: Back up from this first high SNR point to find the earliest point with
#          the *same* polarity.  Take this as i1 = pulse start
    i1 = i0
    s1 = s0
    snr_scale = 1.4
    if s0 and i0:
        for i in range(i0, istart - 3, -1):
            snr = np.abs(data[i]/noise_level)
            #print("i=%d data[i]=%g snr:%f sign[i]=%d --> compare to s0=%d" % (i,data[i],snr,sign[i],s0))
            if sign[i] == s0 and snr >= snr_scale:
                #print("  sign matches --> set i1=i=%d" % i)
                i1 = i
            else:
                #print("  sign NO match --> break")
                break
    #print("s0:%d istart=%d i0=%d i1=%d" % (s0, istart, i0, i1)) 

    '''
# Stage 1: Find at least 6 points moving in the same direction (up or down)

# should be ~5 for OT traces:
    n_in_a_row = 5
    n_in_a_row = 3
    # TODO: for UK data, the pulse width ~ .03 sec/2 = .015 sec - contains about 2 samples!
    n_in_a_row = 2

    #n = int(max_pulse_duration/2. * tr.stats.sampling_rate)
    #n_in_a_row = n

    for i in range(istart, istart + nmax_look):

        abs_diff = np.abs(data[i] - data[i+1])

        print("i:%4d data[i]=%g data[i+1]=%g sign[i]=%d counter=%d" % (i, data[i], data[i+1], sign[i], counter))

        if sign[i] == s:
            counter += 1
        else:
            counter = 0
            s = sign[i]

        if counter == n_in_a_row:
            i1 = i - n_in_a_row
            s1 = sign[i]
            t = tr.stats.starttime + float(i*tr.stats.delta)
            #print("i=%d: t=%s 6th sign in a row [first_sign=%d]" % (i, t, s1))
            break
    '''

    if i1 <= 0:
        print("%s: tr:%s pick_time:%s Didn't pass first test" % (fname, tr.get_id(), pick_time))
        tr.plot()
        return 0, None

    first_sign = s1


# Stage 2: Find the first zero crossing after this
#          And iterate for total of nzeros_to_find subsequent zeros

    zeros = np.array( np.zeros(nzeros_to_find,), dtype=int)
    zeros[0] = i1

    t1 = tr.stats.starttime + float(i1)*tr.stats.delta
    #print("i1=%d --> t=%s" % (i1, t1))

 # MTH: temp hack for UK data:
    #zeros[0] = istart


# TODO: Need to catch flag edge cases where we reach end of range with no zero set!
    for j in range(1,nzeros_to_find):
        #for i in range(i1, i1 + 200):
        for i in range(i1, i1 + nmax_look):
            #print("j=%d i=%d sign=%d" % (j,i,sign[i]))
            if sign[i] != s1:
                #half_per = float( (i - i1) * tr.stats.delta)
                #f = .5 / half_per
                #ipeak,peak = get_peak_amp(tr, i1, i)
                #print("sign:[%2s] t1:%s - t2:%s (T/2:%.6f f:%f) peak:%g" % \
                      #(s1, t1, t2, half_per, f, peak))
                i1 = i
                s1 = sign[i]
                zeros[j] = i
                break
    t1 = tr.stats.starttime + float(zeros[0])*tr.stats.delta
    t2 = tr.stats.starttime + float(zeros[1])*tr.stats.delta

# At this point, first (vel) pulse is located between zeros[0] and zeros[1]
    pulse_width = float(zeros[1]-zeros[0]) * tr.stats.delta
    ipeak, peak_vel = get_peak_amp(tr, zeros[0], zeros[1])
# noise_level defined this way is just for snr comparison
    noise_level = np.max([noise1, noise2, noise3])
    pulse_snr = np.abs(peak_vel / noise_level)

    print("find_zeros: sta:%s cha:%s First pulse t1:%s t2:%s [polarity:%d] pulse_width:%f peak:%g snr:%f" % \
          (tr.stats.station, tr.stats.channel,t1,t2,first_sign, pulse_width, peak_vel, pulse_snr))

# Final gate = try to catch case of early pick on small bump preceding main arrival
#              move istart to end of precursor bump and retry
    if (pulse_width < min_pulse_width or pulse_snr < min_pulse_snr) and not second_try:
        print("Let's RUN THIS ONE AGAIN =============== tr_id:%s" % tr.get_id())
    #if pulse_width < min_pulse_width and not second_try:
        istart = zeros[1]
        return find_signal_zeros(tr, istart, max_pulse_duration=max_pulse_duration, \
                                 nzeros_to_find=nzeros_to_find, second_try=True)

    if debug:
        print("Its here ---> debug is True!!!")
        tr.plot()

    return first_sign, zeros



def get_peak_amp(tr, istart, istop):
    """
    Measure peak (signed) amplitude between istart and istop on trace
    :param tr: velocity trace
    :type tr: obspy.core.trace.Trace or microquake.core.Trace
    :param istart: pick index in trace
    :type istart: int
    :param istop: max index in trace to search
    :type istart: int
    :returns: imax, amp_max: index + value of max
    :rtype: int, float
    """
    abs_max = -1e12
    if istop < istart:
        print("get_peak_amp: istart=%d < istop=%d !" % (istart, istop))
        exit()

    for i in range(istart, istop):
        if np.abs(tr.data[i]) >= abs_max:
            abs_max = np.abs(tr.data[i])
            imax   = i

    return imax, tr.data[imax]



def get_pulse_width_and_area(tr, ipick, icross, max_pulse_duration=.08):
    """
    Measure the width & area of the arrival pulse on the displacement trace
    Start from the displacement peak index (=icross - location of first zero crossing of velocity)

    :param tr: displacement trace
    :type tr: obspy.core.trace.Trace or microquake.core.Trace
    :param ipick: index of pick in trace
    :type ipick: int
    :param icross: index of first zero crossing in corresponding velocity trace
    :type icross: int
    :param max_pulse_duration: max allowed duration (sec) beyond pick to search 
                               for zero crossing of disp pulse
    :type max_pulse_duration: float

    return pulse_width, pulse_area
    :returns: pulse_width, pulse_area: Returns the width and area of the displacement pulse
    :rtype: float, float
    """

    fname = 'get_pulse_width_and_area'

    data = tr.data
    sign = np.sign(data)

    nmax = int(max_pulse_duration * tr.stats.sampling_rate)
    iend = ipick + nmax

    epsilon = 1e-10

    for i in range(icross, iend):
        diff = np.abs(data[i] - data[ipick])
        #print("%d %12.10g %12.10g %d" % (i, data[i], diff, sign[i]))
        if diff < epsilon or sign[i] != sign[icross]:
                break
        if i == iend - 1:
            print("%s: Unable to locate termination of displacement pulse for tr:%s!" % (fname, tr.get_id()))
            return 0, 0

    istop = i
    pulse_width = float(istop - ipick) * tr.stats.delta
    pulse_area  = np.trapz(data[ipick:istop], dx=tr.stats.delta)

    #print("get_pulse_width: t1=%s t2=%s width:%f area:%g" % \
          #(tr.stats.starttime + float(ipick)*tr.stats.delta,
           #tr.stats.starttime + float(istop)*tr.stats.delta,
           #pulse_width, pulse_area))
    #tr.plot()

    return pulse_width, pulse_area

def set_pick_snrs(st, picks, pre_wl=.03, post_wl=.03):
    """
    This function sets the pick snr on each individual trace
     (vs the snr_picker which calculates snr on the composite trace)

    The resulting snr is stored in the tr.stats[key] dict 
    where key = {'P_arrival', 'S_arrival'}

    :param st: traces
    :type st: either obspy.core.Stream or microquake.core.Stream
    :param picks: P & S picks
    :type list: list of either obspy or microquake picks
    :param pre_wl: pre pick window for noise calc
    :type float:
    :param post_wl: post pick window for signal calc
    :type float:
    """

    pick_dict = copy_picks_to_dict(picks)

    for tr in st:
        sta = tr.stats.station
        if sta in pick_dict:
            for phase in pick_dict[sta]:
                pick_time = pick_dict[sta][phase].time
                if phase == 'S':
                    snr = calculate_snr(Stream(traces=[tr]), pick_time, pre_wl=.03, post_wl=.03)
                else:
                    snr = calculate_snr(Stream(traces=[tr]), pick_time, pre_wl=.03, post_wl=.06)
                #print("set snr: tr:%s pha:%s" % (tr.get_id(), phase))
                key = "%s_arrival" % phase
                if key not in tr.stats:
                    tr.stats[key] = {}
                tr.stats[key]['snr'] = snr
        else:
            print("set_pick_snrs: sta:%s not in pick_dict" % sta)

    return


def get_arrival_from_pick(arrivals, pick):
    arrival = None
    for arr in arrivals:
        if arr.pick_id == pick.resource_id:
            arrival = arr
            break

    return arrival
