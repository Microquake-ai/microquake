
""" Waveform amplitude measurements

    This module contains a collection of functions for making
    measurements on the velocity and displacement waveforms
    that are later used to calculate moment magnitude, focal mechanism, etc

1234567890123456789012345678901234567890123456789012345678901234567890123456789
"""

import numpy as np

from microquake.core.util.tools import copy_picks_to_dict
from microquake.core.stream import Stream
from microquake.waveform.pick import calculate_snr
from microquake.core.data.inventory import get_sensor_type_from_trace
from microquake.core.event import get_arrival_from_pick

# default logger
import logging
logger = logging.getLogger(__name__)

def measure_pick_amps(st_in, cat, phase_list=None,
                      logger_in=None, triaxial_only=False,
                      **kwargs):

    """
    Attempt to measure velocity pulse parameters (polarity, peak vel, etc)
      and displacement pulse parameters (pulse width, area) 
      for each arrival for each event preferred_origin in cat

    Measures are made on individual traces, saved to arrival.traces[trace id],
      and later combined to one measurement per arrival
      and added to the *arrival* extras dict

    :param st_in: velocity traces
    :type st_in: obspy.core.Stream or microquake.core.Stream
    :param cat: obspy.core.event.Catalog
    :type cat: list of obspy.core.event.Events or microquake.core.event.Events
    :param phase_list: ['P'], ['S'], or ['P', 'S'] - list of arrival phases to process
    :type phase_list: list
    :param triaxial_only: if True --> only keep 3-component observations (disp area) in arrival dict
    :type triaxial_only: boolean
    """

    fname = "measure_pick_amps"

    global logger
    if logger_in is not None:
        logger = logger_in

    st = st_in.copy()

    measure_velocity_pulse(st, cat, phase_list=phase_list, **kwargs)

    debug = False
    if 'debug' in kwargs:
        debug = kwargs['debug']

    measure_displacement_pulse(st, cat, phase_list=phase_list, debug=debug)

    # Combine individual trace measurements (peak_vel, dis_pulse_area, etc)
    #    into one measurement per arrival:

    for event in cat:
        for phase in phase_list:
            origin = event.preferred_origin() if event.preferred_origin() else event.origins[0]
            arrivals = sorted([x for x in origin.arrivals if x.phase == phase],
                            key=lambda x: int(x.get_pick().get_sta()) )

            for arr in arrivals:

                pk=arr.get_pick()
                sta=pk.get_sta()

                if arr.traces is not None:
                    dis_area = []
                    dis_width = []

                    for tr_id,v in arr.traces.items():
                        #print("arr: sta:%s [%s] tr:%s pol:%d peak_vel:%s disp_area:%s" % \
                            #(sta, arr.phase, tr_id, v['polarity'], v['peak_vel'], v['dis_pulse_area']))

            # For now, since this is primarily going to be used for P-arrivals
            #   Set arrival values from the vertical (Z) or P (if rotated) component:
                        if v['polarity'] != 0 and tr_id[-1].upper() in ['Z', 'P']:
                            #print("Set polarity=%d from tr:%s" % (v['polarity'], tr_id))
                            arr.polarity = v['polarity']
                            arr.t1 = v['t1']
                            arr.t2 = v['t2']
                            arr.peak_vel = v['peak_vel']
                            arr.tpeak_vel = v['tpeak_vel']
                            arr.pulse_snr = v['pulse_snr']

                        if v['peak_dis'] != None and tr_id[-1].upper() in ['Z', 'P']:
                            arr.peak_dis = v['peak_dis']
                            arr.max_dis = v['max_dis']
                            arr.tpeak_dis = v['tpeak_dis']
                            arr.tmax_dis = v['tmax_dis']

            # But average vector quantities distributed over components:
                        if v['dis_pulse_area'] is not None:
                            dis_area.append(v['dis_pulse_area'])
                        if v['dis_pulse_width'] is not None:
                            dis_width.append(v['dis_pulse_width'])

            # Here is where you could impose triaxial_only requirement
            #  but this will filter out not only 1-chan stations, but
            #  any stations where peak finder did not locate peak on
            #  *all* 3 channels
                    #if triaxial_only and len(dis_area) == 3:

                    if triaxial_only and len(dis_area) == 3:
                        arr.dis_pulse_area = np.sqrt(np.sum(np.array(dis_area)**2))
                    elif len(dis_area) > 0:
                        arr.dis_pulse_area = np.sqrt(np.sum(np.array(dis_area)**2))

                    if len(dis_width) > 0:
                        arr.dis_pulse_width =  np.mean(dis_width)

                    #print()
                else:
                    #print("sta:%3s pha:%s pk.time:%s --> Has no traces dict ****" % (sta, arr.phase, pk.time))
                    pass

    return


def measure_velocity_pulse(st,
                           cat,
                           phase_list=None,
                           pulse_min_width=.02,
                           pulse_min_snr_P=7,
                           pulse_min_snr_S=5,
                           debug=False,
                           ):
    """
    locate velocity pulse (zero crossings) near pick and measure peak amp,
        polarity, etc on it

    All measurements are added to the *arrival* extras dict

    :param st: velocity traces
    :type st: obspy.core.Stream or microquake.core.Stream
    :param cat: obspy.core.event.Catalog
    :type cat: list of obspy.core.event.Events or microquake.core.event.Events
    :param phase_list: ['P'], ['S'], or ['P', 'S'] - list of arrival phases to process
    :type phase_list: list
    :param pulse_min_width: Measured first pulse must be this wide to be retained
    :type pulse_min_width: float
    :param pulse_min_snr_P: Measure first P pulse must have snr greater than this
    :type pulse_min_snr_P: float
    :param pulse_min_snr_S: Measure first S pulse must have snr greater than this
    :type pulse_min_snr_S: float
    """

    fname = 'measure_velocity_pulse'

    if phase_list is None:
        phase_list = ['P']

    # Average of P,S min snr used for finding zeros
    min_pulse_snr = int((pulse_min_snr_P + pulse_min_snr_S)/2)

    for event in cat:
        origin = event.preferred_origin() if event.preferred_origin() else event.origins[0]
        arrivals = origin.arrivals

        for arr in arrivals:

            phase = arr.phase

            if phase not in phase_list:
                continue

            pk  = arr.get_pick()
            if pk is None:
                logger.error("%s: arr pha:%s id:%s --> Lost reference to pick id:%s --> SKIP" % \
                             (fname, arr.phase, arr.resource_id.id, arr.pick_id.id))
                continue
            sta = pk.get_sta()

            trs = st.select(station=sta)

            if trs is None:
                logger.warn("%s: sta:%s has a [%s] arrival but no trace in stream --> Skip" % \
                            (fname, sta, arr.phase))
                continue

            sensor_type = get_sensor_type_from_trace(trs[0])

            if sensor_type != "VEL":
                logger.info("%s: sta:%s sensor_type != VEL --> Skip" % (fname, sta))
                continue

            arr.traces = {}

            for tr in trs:
                try:
                    tr.detrend("demean").detrend("linear")
                except Exception as e:
                    print(e)
                    continue
                data = tr.data.copy()
                ipick = int((pk.time - tr.stats.starttime) * tr.stats.sampling_rate)

                polarity, vel_zeros = _find_signal_zeros(
                                               tr, ipick,
                                               nzeros_to_find=3,
                                               min_pulse_width=pulse_min_width,
                                               min_pulse_snr=min_pulse_snr,
                                               debug=debug
                                               )

                dd = {}
                #dd['pick_time'] = pk.time
                dd['polarity'] = 0
                dd['t1'] = None
                dd['t2'] = None
                dd['peak_vel'] = None
                dd['tpeak_vel'] = None
                dd['pulse_snr'] = None

                # A good pick will have the first velocity pulse located
                #    between i1 and i2
                if vel_zeros is not None:
                    i1 = vel_zeros[0]
                    i2 = vel_zeros[1]
                    t1 = tr.stats.starttime + float(i1 * tr.stats.delta)
                    t2 = tr.stats.starttime + float(i2 * tr.stats.delta)

                    ipeak, peak_vel = _get_peak_amp(tr, i1, i2)
                    tpeak = tr.stats.starttime + float(ipeak * tr.stats.delta)

                    noise_npts = int(.01 * tr.stats.sampling_rate)
                    noise_end = ipick - int(.005 * tr.stats.sampling_rate)
                    noise = data[noise_end - noise_npts: noise_end]
                    noise1 = np.abs(np.mean(noise))
                    noise2 = np.abs(np.median(noise))
                    noise3 = np.abs(np.std(noise))
                    noise_level = np.max([noise1, noise2, noise3])

                    pulse_snr = np.abs(peak_vel/noise_level)
                    pulse_width = float((i2-i1)*tr.stats.delta)

                    pulse_thresh = pulse_min_snr_P

                    if phase == 'S':
                        pulse_thresh = pulse_min_snr_S

                    if pulse_snr < pulse_thresh:
                        logger.debug("%s: tr:%s pha:%s t1:%s t2:%s pulse_snr=%.1f \
                               < thresh" % \
                               (fname, tr.get_id(), phase, t1, t2, pulse_snr))
                        polarity = 0

                    if pulse_width < pulse_min_width:
                        logger.debug("%s: tr:%s pha:%s t1:%s t2:%s pulse_width=%f \
                               < .0014" % \
                             (fname, tr.get_id(), phase, t1, t2, pulse_width))
                        polarity = 0


                    dd['polarity'] = polarity
                    dd['peak_vel'] = peak_vel
                    dd['tpeak_vel'] = tpeak
                    dd['t1'] = t1
                    dd['t2'] = t2
                    dd['pulse_snr'] = pulse_snr

                else:
                    logger.debug("%s: Unable to locate zeros for tr:%s pha:%s" % \
                          (fname, tr.get_id(), phase))

                arr.traces[tr.get_id()] = dd

            # Process next phase in phase_list

        # Process tr in st

    # Process next event in cat

    return


def measure_displacement_pulse(st,
                               cat,
                               phase_list=None,
                               debug=False):
    """
    measure displacement pulse (area + width) for each pick on each arrival
        as needed for moment magnitude calculation

    All measurements are added to the *arrival* extras dict

    :param st: velocity traces
    :type st: obspy.core.Stream or microquake.core.Stream
    :param cat: obspy.core.event.Catalog
    :type list: list of obspy.core.event.Events or microquake.core.event.Events
    """

    fname = 'measure_displacement_pulse'

    if phase_list is None:
        phase_list = ['P']

    for event in cat:
        origin = event.preferred_origin() if event.preferred_origin() else event.origins[0]
        arrivals = origin.arrivals

        for arr in arrivals:

            phase = arr.phase

            if phase not in phase_list:
                continue

            pk  = arr.get_pick()
            if pk is None:
                logger.error("%s: arr pha:%s id:%s --> Lost reference to pick id:%s --> SKIP" % \
                             (fname, arr.phase, arr.resource_id.id, arr.pick_id.id))
                continue
            sta = pk.get_sta()

            trs = st.select(station=sta)

            if trs is None:
                logger.warn("%s: sta:%s has a [%s] arrival but no trace in stream --> Skip" % \
                            (fname, sta, arr.phase))
                continue

            sensor_type = get_sensor_type_from_trace(trs[0])

            if sensor_type != "VEL":
                logger.info("%s: sta:%s sensor_type != VEL --> Skip" % (fname, sta))
                continue


            for tr in trs:
                try:
                    tr_dis = tr.copy().detrend("demean").detrend("linear")
                    tr_dis.integrate().detrend("linear")
                except Exception as e:
                    print(e)
                    continue
                tr_dis.stats.channel = "%s.dis" % tr.stats.channel

                dd = {}
                dd['peak_dis'] = None
                dd['max_dis'] = None
                dd['tpeak_dis'] = None
                dd['tmax_dis'] = None
                dd['dis_pulse_width'] = None
                dd['dis_pulse_area'] = None

                tr_dict = arr.traces[tr.get_id()]

                polarity = tr_dict['polarity']
                t1 = tr_dict.get('t1', None)
                t2 = tr_dict.get('t2', None)

                #print("tr:%s pol:%d t1:%s t2:%s" % (tr.get_id(), polarity, t1, t2))

                if polarity != 0:

                    if t1 is None or t2 is None:
                        logger.error("%s: t1 or t2 is None --> You shouldn't be here!" \
                                     % (fname))
                        continue

                    i1 = int((t1 - tr.stats.starttime) * tr.stats.sampling_rate)
                    i2 = int((t2 - tr.stats.starttime) * tr.stats.sampling_rate)

                    ipick = int((pk.time - tr.stats.starttime) * tr.stats.sampling_rate)

                    icross = i2
                    tr_dis.data = tr_dis.data - tr_dis.data[i1]
                    #tr_dis.data = tr_dis.data - tr_dis.data[ipick]

                    dis_polarity = np.sign(tr_dis.data[icross])
                    pulse_width, pulse_area = _get_pulse_width_and_area(tr_dis, i1, icross)

                    npulse = int(pulse_width * tr.stats.sampling_rate)

                    max_pulse_duration = .08
                    nmax_len = int(max_pulse_duration * tr.stats.sampling_rate)

                    if pulse_width != 0:

                        ipeak, peak_dis = _get_peak_amp(tr_dis, ipick,
                                                        ipick + npulse)
                        # max_dis = max within max_pulse_duration of pick time
                        imax, max_dis = _get_peak_amp(tr_dis, ipick,
                                                        ipick + nmax_len)

                        tmax_dis = tr.stats.starttime + float(imax * tr.stats.delta)
                        tpeak_dis = tr.stats.starttime + float(ipeak * tr.stats.delta)
                        tcross_dis = pk.time + pulse_width

                        dd['peak_dis'] = peak_dis
                        dd['max_dis'] = max_dis
                        dd['tpeak_dis'] = tpeak_dis
                        dd['tmax_dis'] = tmax_dis
                        dd['dis_pulse_width'] = pulse_width
                        dd['dis_pulse_area'] = pulse_area


                        if debug:
                            logger.debug("[%s] Dis pol=%d tpick=%s" % \
                                (phase, dis_polarity, pk.time))
                            logger.debug("              tpeak=%s peak_dis=%12.10g" %
                                (tpeak_dis, peak_dis))
                            logger.debug("             tcross=%s" % tcross_dis)
                            logger.debug("               tmax=%s max_dis=%12.10g" %
                                (tmax_dis, max_dis))
                            logger.debug("    dis pulse width=%.5f" % pulse_width)
                            logger.debug("    dis pulse  area=%12.10g" % pulse_area)

                    else:
                        logger.warn("%s: Got pulse_width=0 for tr:%s pha:%s" %
                                    (fname, tr.get_id(), phase))

                arr.traces[tr.get_id()] = dict(tr_dict, **dd)

            # Process next tr in trs

        # Process next arr in arrivals

    # Process next event in cat

    return


def _find_signal_zeros(tr, istart, max_pulse_duration=.1, nzeros_to_find=3,
                       second_try=False, debug=False,
                       min_pulse_width=.00167, min_pulse_snr=5):

    """
    Locate zero crossing of velocity trace to locate first pulse(s)
    All measurements are added to the *arrival* extras dict

    :param tr: Individual velocity trace
    :type tr: obspy.core.Trace or microquake.core.Trace
    :param istart: index in trace.data to start searching from (eg =pick index)
    :type istart: int
    :param max_pulse_duration: maximum search window (in seconds) for end
                               of pulse
    :type max_pulse_duration: float
    :param nzeros_to_find: Number of zero crossings to find
    :type nzeros_to_find: int
    :param second_try: If true then iterate once over early/late pick to locate
                       first pulse
    :type second_try: boolean
    :param min_pulse_width: minimum allowable pulse width (in seconds)
    :type min_pulse_width: float
    :param min_pulse_snr: minimum allowable pulse snr
    :type min_pulse_snr: float
    :param debug: If true then output/plot debug msgs
    :type debug: boolean

    :returns: first_sign, zeros
    :rtype: int, np.array
    """

    fname = '_find_signal_zeros'

    data = tr.data
    sign = np.sign(data)

    i1 = -9

    if second_try:
        logger.debug("%s: This is the second try!" % fname)

    noise_tlen = .05
    noise_npts = int(noise_tlen * tr.stats.sampling_rate)
    noise_end = istart - int(.005 * tr.stats.sampling_rate)
    #noise_level = np.mean(data[noise_end - noise_npts: noise_end])
    noise = data[noise_end - noise_npts: noise_end]
    noise1 = np.abs(np.mean(noise))
    noise2 = np.abs(np.median(noise))
    noise3 = np.abs(np.std(noise))
    noise_level = np.max([noise1, noise2, noise3])
    noise_level = noise1

    #pick_snr = np.abs(data[istart]/noise_level)

    nmax_look = int(max_pulse_duration * tr.stats.sampling_rate)

    # Just used for debug
    pick_time = tr.stats.starttime + float(istart*tr.stats.delta)

# Stage 0: Take polarity sign (s0) from first data point after
#          after istart (=ipick) with SNR >= thresh * noise_level

    s0 = 0
    i0 = 0
    snr_thresh = 10.
    for i in range(istart, istart + nmax_look):
        if np.abs(data[i]) >= snr_thresh * np.abs(noise_level):
            s0 = sign[i]
            i0 = i
            break

# Stage 1: Back up from this first high SNR point to find the earliest point
#          with the *same* polarity.  Take this as i1 = pulse start
    i1 = i0
    s1 = s0
    snr_scale = 1.4
    if s0 and i0:
        for i in range(i0, istart - 3, -1):
            snr = np.abs(data[i]/noise_level)
            if sign[i] == s0 and snr >= snr_scale:
                #print("  sign matches --> set i1=i=%d" % i)
                i1 = i
            else:
                #print("  sign NO match --> break")
                break

    if i1 <= 0:
        logger.debug("%s: tr:%s pick_time:%s Didn't pass first test" %
              (fname, tr.get_id(), pick_time))
        #tr.plot()
        return 0, None

    first_sign = s1


# Stage 2: Find the first zero crossing after this
#          And iterate for total of nzeros_to_find subsequent zeros

    zeros = np.array(np.zeros(nzeros_to_find,), dtype=int)
    zeros[0] = i1

    t1 = tr.stats.starttime + float(i1)*tr.stats.delta
    #print("i1=%d --> t=%s" % (i1, t1))


# TODO: Need to catch flag edge cases where we reach end of range with
#       no zero set!
    for j in range(1, nzeros_to_find):
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
    ipeak, peak_vel = _get_peak_amp(tr, zeros[0], zeros[1])
# noise_level defined this way is just for snr comparison
    noise_level = np.max([noise1, noise2, noise3])
    pulse_snr = np.abs(peak_vel / noise_level)

    logger.debug("find_zeros: sta:%s cha:%s First pulse t1:%s t2:%s [polarity:%d] \
           pulse_width:%f peak:%g snr:%f" % \
           (tr.stats.station, tr.stats.channel, t1, t2, first_sign,
            pulse_width, peak_vel, pulse_snr))

# Final gate = try to catch case of early pick on small bump preceding main
#              arrival move istart to end of precursor bump and retry
    if ((pulse_width < min_pulse_width or pulse_snr < min_pulse_snr)
            and not second_try):

        logger.debug("Let's RUN THIS ONE AGAIN ============== tr_id:%s" % tr.get_id())
    #if pulse_width < min_pulse_width and not second_try:
        istart = zeros[1]
        return _find_signal_zeros(tr, istart,
                                  max_pulse_duration=max_pulse_duration,
                                  nzeros_to_find=nzeros_to_find,
                                  second_try=True)

    #if debug:
        #tr.plot()

    return first_sign, zeros


def _get_peak_amp(tr, istart, istop):
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
        logger.error("_get_peak_amp: istart=%d < istop=%d !" % (istart, istop))
        exit()

    for i in range(istart, istop):
        if np.abs(tr.data[i]) >= abs_max:
            abs_max = np.abs(tr.data[i])
            imax = i

    return imax, tr.data[imax]


def _get_pulse_width_and_area(tr, ipick, icross, max_pulse_duration=.08):
    """
    Measure the width & area of the arrival pulse on the displacement trace
    Start from the displacement peak index (=icross - location of first zero
            crossing of velocity)

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
    :returns: pulse_width, pulse_area: Returns the width and area of the
                                       displacement pulse
    :rtype: float, float
    """

    fname = '_get_pulse_width_and_area'

    data = tr.data
    sign = np.sign(data)

    nmax = int(max_pulse_duration * tr.stats.sampling_rate)
    iend = ipick + nmax

    epsilon = 1e-10

    i = 0 # Just to shut pylint up
    for i in range(icross, iend):
        diff = np.abs(data[i] - data[ipick])
        #print("%d %12.10g %12.10g %d" % (i, data[i], diff, sign[i]))
        if diff < epsilon or sign[i] != sign[icross]:
            break
        if i == iend - 1:
            logger.info("%s: Unable to locate termination of displacement pulse for \
                  tr:%s!" % (fname, tr.get_id()))
            return 0, 0

    istop = i
    pulse_width = float(istop - ipick) * tr.stats.delta
    pulse_area = np.trapz(data[ipick:istop], dx=tr.stats.delta)

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
                    snr = calculate_snr(Stream(traces=[tr]), pick_time,
                                        pre_wl=pre_wl, post_wl=post_wl)
                else:
                    snr = calculate_snr(Stream(traces=[tr]), pick_time,
                                        pre_wl=pre_wl, post_wl=post_wl)
                #print("set snr: tr:%s pha:%s" % (tr.get_id(), phase))
                key = "%s_arrival" % phase
                if key not in tr.stats:
                    tr.stats[key] = {}
                tr.stats[key]['snr'] = snr
        else:
            logger.warn("set_pick_snrs: sta:%s not in pick_dict" % sta)

    return



from scipy.fftpack import rfft
#from microquake.waveform.smom_mag_utils import npow2, unpack_rfft
from microquake.waveform.parseval_utils import npow2, unpack_rfft

def calc_velocity_flux(st_in,
                       cat,
                       phase_list=None,
                       use_fixed_window=True,
                       pre_P=.01,
                       P_len=.05,
                       pre_S=.01,
                       S_len=.1,
                       Q=1e12,
                       correct_attenuation=False,
                       triaxial_only=True,
                       debug=False, logger_in=logger):
    """
    For each arrival (on phase_list) calculate the velocity flux using
        the corresponding traces and save to the arrival.vel_flux to
        be used in the calculation of radiated seismic energy

    :param st_in: velocity traces
    :type st_in: obspy.core.Stream or microquake.core.Stream
    :param cat: obspy.core.event.Catalog
    :type cat: list of obspy.core.event.Events or microquake.core.event.Events
    :param phase_list: ['P'], ['S'], or ['P', 'S'] - list of arrival phases to process
    :type phase_list: list
    :param triaxial_only: if True --> only calc flux for 3-comp stations
    :type triaxial_only: boolean
    :param Q: Anelastic Q to use for attenuation correction to flux
    :type Q: float
    :param correct_attenuation: if True, scale spec by e^-pi*f*travel-time/Q before summing
    :type correct_attenuation: boolean
    """


    fname = "calc_velocity_flux"

    global logger
    if logger_in is not None:
        logger = logger_in

    if phase_list is None:
        phase_list = ['P']

    #MTH: Try to avoid copying cat - it causes the events to lose their link to preferred_origin!
    #cat = cat_in.copy()
    # Defensive copy - not currently needed since we only trim on copies, still ...
    st = st_in.copy().detrend('demean').detrend('linear')

    for event in cat:
        origin = event.preferred_origin() if event.preferred_origin() else event.origins[0]

        for arr in origin.arrivals:

            phase = arr.phase
            if phase not in phase_list:
                continue

            pick = arr.get_pick()
            if pick is None:
                logger.error("%s: arr pha:%s id:%s --> Lost reference to pick id:%s --> SKIP" % \
                             (fname, arr.phase, arr.resource_id.id, arr.pick_id.id))
                continue
            sta  = pick.get_sta()

            trs = st.select(station=sta)

            if trs is None:
                logger.warn("%s: sta:%s has a [%s] arrival but no trace in stream --> Skip" % \
                            (fname, sta, phase))
                continue

            if triaxial_only and len(trs) != 3:
                logger.info("%s: sta:%s is not 3-comp --> Skip" % (fname, sta))
                continue

            sensor_type = get_sensor_type_from_trace(trs[0])

            if sensor_type != "VEL":
                logger.info("%s: sta:%s sensor_type != VEL --> Skip" % (fname, sta))
                continue


            if use_fixed_window:
                if phase == 'P':
                    pre = pre_P
                    win_secs = P_len
                else:
                    pre = pre_S
                    win_secs = S_len

                starttime = pick.time - pre
                endtime = starttime + win_secs

            not_enough_trace = False

            for tr in trs:
                if starttime < tr.stats.starttime or endtime > tr.stats.endtime:
                    logger.warn("%s: sta:%s pha:%s tr:%s is too short to trim --> Don't use" % \
                                (fname, sta, phase, tr.get_id()))
                    not_enough_trace = True
                    break

            if not_enough_trace:
                continue

            tr3 = trs.copy()

            tr3.trim(starttime=starttime, endtime=endtime)
            dt= tr3[0].stats.delta

            #flux_t = np.sum( [tr.data**2 for tr in tr3]) * dt

            fluxes = []
            for tr in tr3:
                tsum = np.sum(tr.data**2)*dt
                fluxes.append(tsum**2)

            flux_t = np.sqrt(np.sum(fluxes))

            if not correct_attenuation:
                flux = flux_t

        # The only reason to do this in the freq domain is if we
        #    want to apply attenuation correction
            else:
                #logger.info("%s: Correcting for Attenuation [Q=%f]" % (fname, Q))
                # travel_time: from pick_time - origin or from R/v where v={alpha,beta}
                travel_time = pick.time - origin.time

                flux_f = 0.
                fluxes = []
                for tr in tr3:
                    data = tr.data
                    nfft = npow2(data.size)
                    df = 1./(dt * float(nfft))    # df is same as for 2-sided case
                    Y,freqs = unpack_rfft(rfft(data, n=nfft), df)
                    Y *= dt
                    # 1-sided: N/2 -1 +ve freqs + [DC + Nyq] = N/2 + 1 values:
                    Y[1:-1] *= np.sqrt(2.)
                    # Correct for attenuation: For testing making Q=1e12 so that flux_f = flux_t by Parseval's
                    x = 0.6
                    Qf = Q*freqs**x
                    tstar = travel_time / Qf
                    tstar[0] = 0.

                    #fsum = np.sum( np.abs(Y)*np.abs(Y) )*df
                    #fsum = np.sum( np.abs(Y)*np.abs(Y) * np.exp(2.*np.pi*freqs*tstar))*df

                    fsum = np.sum(np.abs(Y)*np.abs(Y)*np.exp(2.*np.pi*freqs*tstar)) * df

                    tsum = np.sum(tr.data**2)*dt
                    #print("id:%s flux_f:%g flux_t:%g" % (tr.get_id(), fsum, tsum))

                    D = np.array( np.zeros(freqs.size), dtype=np.float_)
                    for i,f in enumerate(freqs):
                        D[i] = np.sum(np.abs(Y[:i]**2)) * df

                    title = "%s [%s]" % (tr.get_id(), phase)
                    #plot_Df(freqs, D, title=title)

                    fluxes.append(fsum**2)

                    flux_f += fsum

                flux_f = np.sqrt(np.sum(fluxes))

                flux = flux_f

                #print("Parseval's: nfft=%7d df=%12.6g flux_f=%12.10g flux_t=%12.10g" % \
                      #(nfft, df, fsum, flux_t))

                #print("sta:%3s [%s] R:%.1f flux_t:%g flux_f:%g" % (sta, phase, arr.distance, flux_t, flux_f))
                #exit()

            arr.vel_flux = flux

    return

import matplotlib.pyplot as plt

def plot_Df(freqs, D, title=None):

    plt.plot(freqs, D, color='blue')

    #plt.loglog(freqs, model_spec,  color='green')
    #plt.legend(['signal', 'model'])
    #plt.xlim(1e0, 3e3)
    #plt.ylim(1e-12, 1e-4)
    if title:
        plt.title(title)
    plt.grid()
    plt.show()

    #exit()

    return

    #foo = sorted([x for x in S_P_times], key=lambda x: x[2])



