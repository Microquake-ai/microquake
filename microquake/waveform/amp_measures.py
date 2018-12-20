import numpy as np

from microquake.core.util.tools import copy_picks_to_dict
from microquake.waveform.helpers import *
from microquake.core.stream import Stream
from microquake.waveform.pick import calculate_snr

""" This module's docstring summary line.
    This is a multi-line docstring. Paragraphs are separated with blank lines.

    Lines conform to 79-column limit.

    Module and packages names should be short, lower_case_with_underscores.

         1         2         3         4         5         6         7
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""

def set_pick_snrs(st, picks, pre_wl=.03, post_wl=.03):

    pick_dict = copy_picks_to_dict(picks)

    for tr in st:
        sta = tr.stats.station
        if sta in pick_dict:
            for phase in pick_dict[tr.stats.station]:
                pick_time = pick_dict[tr.stats.station][phase].time
                snr = calculate_snr(Stream(traces=[tr]), pick_time, pre_wl=.03, post_wl=.03)
                key = "%s_arrival" % phase
            #print("tr:%s pha:%s time:%s snr:%.1f key=%s" % (tr.get_id(), phase, pick_time, snr, key))
                if key not in tr.stats:
                    tr.stats[key] = {}
                tr.stats[key]['snr'] = snr
        else:
            print("set_pick_snrs: sta:%s not in pick_dict" % sta)

    return

def measure_pick_amps(st, picks, debug=False):

    """
    For each tr in st, for each pick phase (P,S) in picks:
        Measures the polarity and zero crossing from velocity trace
        Measures the pulse width and area from displacement trace
        Measures the arrival peak amps and times for acc, vel, disp 

    All measurements are added to the stats dict of each trace in the stream
        e.g., tr.stats.P_arrival.polarity
              tr.stats.P_arrival.peak_vel
              tr.stats.P_arrival.tpeak
                    ...
              tr.stats.P_arrival.dis_pulse_width
              tr.stats.P_arrival.dis_pulse_area

    :param st: velocity traces
    :type st: either obspy.core.Stream or microquake.core.Stream
    :param picks: P & S picks
    :type list: list of either obspy or microquake picks
    """

    # TODO: some of the sites have accelerometers and need to be integrated first!

    fname = 'measure_pick_amps'

    #st2 = Stream()
    #for sta in ['25','58','90','41', '25', '31', '28', '58', '12', '23', '15', '59', '89']:
    #for sta in ['25', '40', '41']:
        #st2 += st.select(station=sta)

    measure_velocity_pulse(st, picks, debug=False)

    #measure_displacement_pulse(st2, picks, debug=False)

    return


def measure_velocity_pulse(st, picks, debug=False):

    # TODO: Need to pass in metadata and check for accel, or filter these out beforehand

    fname = 'measure_velocity_pulse'

    pick_dict = copy_picks_to_dict(picks)

    for tr in st:

        sta = tr.stats.station

        tr.detrend("demean").detrend("linear")

        data = tr.data.copy()

        snr_thresh = 3.0

        for phase in ['P']:
        #for phase in ['P', 'S']:

            if sta in pick_dict and phase in pick_dict[sta]:
                pass
            else:
                print("%s: sta:%s has no [%s] pick" % (fname, sta, phase))
                continue

            key = "%s_arrival" % phase

            snr = tr.stats[key]['snr']

            if snr < snr_thresh:
                print("tr:%s pha:%s snr:%.1f < thresh --> Skip" % (tr.get_id(), phase, snr))
                #tr.plot()
                continue

            pick_time = pick_dict[sta][phase].time

            ipick = int((pick_time - tr.stats.starttime) * tr.stats.sampling_rate)

            polarity, vel_zeros = find_signal_zeros(tr, ipick, nzeros_to_find=3)

            dd = {}
            dd['pick_time']= pick_time

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
                noise_level = np.std(data[noise_end - noise_npts: noise_end])

                pulse_snr = np.abs(peak_vel/noise_level)
                pulse_width = float((i2-i1)*tr.stats.delta)

                if pulse_snr < 9. or pulse_width < .0014:
                    polarity = 0

                dd['polarity'] = polarity
                dd['peak_vel'] = peak_vel
                dd['tpeak'] = tpeak
                dd['t1'] = t1
                dd['t2'] = t2
                dd['pulse_snr'] = pulse_snr
            #dd['vel_period']  = vel_period

            else:
                print("Unable to locate zeros for tr:%s pha:%s" % (tr.get_id(),phase))
                polarity = 0
                dd['polarity'] = polarity


            tr.stats[key]['velocity_pulse'] = dd

    return



def measure_displacement_pulse(st, picks, debug=False):

    fname = 'measure_displacement_pulse'

    pick_dict = copy_picks_to_dict(picks)

    #plot = True
    plot = False

    for tr in st:

        sta = tr.stats.station


        tr_dis = tr.copy().detrend("demean").detrend("linear")
        tr_dis.integrate().detrend("linear")
        tr_dis.stats.channel="%s.dis" % tr.stats.channel

        snr_thresh = 3.0

        for phase in ['P']:
        #for phase in ['P', 'S']:

            key = "%s_arrival" % phase

            snr = tr.stats[key]['snr']

            if snr < snr_thresh:
                print("tr:%s pha:%s snr:%.1f < thresh --> Skip" % (tr.get_id(), phase, snr))
                tr.plot()
                continue

            polarity = tr.stats[key]['velocity_pulse']['polarity']
            t1 = tr.stats[key]['velocity_pulse']['t1']
            t2 = tr.stats[key]['velocity_pulse']['t2']

            pick_time = pick_dict[sta][phase].time

            i1 = int((t1 - tr.stats.starttime) * tr.stats.sampling_rate)
            i2 = int((t2 - tr.stats.starttime) * tr.stats.sampling_rate)

            ipick = int((pick_time - tr.stats.starttime) * tr.stats.sampling_rate)

            dd = {}


            if polarity != 0:

                icross = i2
                tr_dis.data = tr_dis.data - tr_dis.data[i1]
                #tr_dis.data = tr_dis.data - tr_dis.data[ipick]

                dis_polarity = np.sign(tr_dis.data[icross])
                pulse_width, pulse_area = measure_displacement_pulse(tr_dis, i1, icross)
                #pulse_width, pulse_area = measure_displacement_pulse(tr_dis, ipick, icross)

                npulse = int(pulse_width * tr.stats.sampling_rate)

                if pulse_width != 0:

                    ipeak,peak_dis = get_peak_amp(tr_dis, ipick, ipick + npulse)
                    imax, max_dis  = get_peak_amp(tr_dis, ipick, ipick + nmax_len)

                    tmax_dis  = tr.stats.starttime + float(imax * tr.stats.delta)
                    tpeak_dis = tr.stats.starttime + float(ipeak * tr.stats.delta)
                    tcross_dis = pick_time + pulse_width

                    dd['peak_dis'] = peak_dis
                    dd['max_dis']  = max_dis
                    dd['tpeak_dis'] = tpeak_dis
                    dd['tmax_dis']  = tmax_dis
                    dd['dis_pulse_width'] = pulse_width
                    dd['dis_pulse_area']  = pulse_area

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

            #if key in tr.stats:
                #arrival_dict = tr.stats[key]
                #for k,v in tr.stats[key].items():
                    #print("Found k=%s --> v=%s" % (k,v))
            #else:
            #exit()

    # TODO Need to check so we don't overwrite P_arrival/S_arrival dict if already created
            #if key in tr.stats:
                #tr.stats[key]

            tr.stats[key]['displacement_pulse'] = dd

            return



#def find_signal_zeros(tr, istart, max_pulse_duration=.08, nzeros_to_find=3, 
                      #walk_back_pick=False, min_noise_level=0):

def find_signal_zeros(tr, istart, max_pulse_duration=.08, nzeros_to_find=3):

    fname = 'find_signal_zeros'

    data = tr.data
    sign = np.sign(data)

    counter = 0
    s = 0
    i1 = -9

    #noise_npts  = int(.01 * tr.stats.sampling_rate)
    #noise_end   = istart - int(.005 * tr.stats.sampling_rate)
    #noise_level = np.std(data[noise_end - noise_npts: noise_end])

# Stage 1: Find at least 6 points moving in the same direction (up or down)

    scale = 1.4
    for i in range(istart, istart + 100):

        abs_diff = np.abs(data[i] - data[i+1])

        #if sign[i] == s and abs_diff > scale * noise_level:
            #print("increment counter: abs_diff=%g noise_level=%g" % (abs_diff, noise_level))
        if sign[i] == s:
            counter += 1
        else:
            counter = 0
            s = sign[i]

        if counter == 5:
            i1 = i - 5
            s1 = sign[i]
            t = tr.stats.starttime + float(i*tr.stats.delta)
            #print("i=%d: t=%s 6th sign in a row [first_sign=%d]" % (i, t, s1))
            break

    if i1 < 0:
        print("%s: tr:%s Didn't pass first test" % (fname, tr.get_id()))
        return 0, None

    first_sign = s1

# Stage 2: Find the first zero crossing after this
# Could either check for polarity flip (assuming zero mean trace) or abs(amp) back to pick value

    zeros = np.array( np.zeros(nzeros_to_find,), dtype=int)
    zeros[0] = i1

    for j in range(1,nzeros_to_find):
        for i in range(i1, i1 + 100):
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



def measure_displacement_pulse(tr, ipick, icross, max_pulse_duration=.08):
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

    fname = 'measure_displacement_pulse'

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

    return pulse_width, pulse_area


