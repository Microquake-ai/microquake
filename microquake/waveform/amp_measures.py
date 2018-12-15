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

    pick_dict = copy_picks_to_dict(picks)

    sta = '41'
    #sta = '15'
    #sta = '31'
    sta = '12'
    sta = '28'

    st2 = st.select(station=sta)
    #plot = True
    plot = False

    #for tr in st2:
    for tr in st:

        sta = tr.stats.station

        tr.detrend("demean").detrend("linear")

        data = tr.data.copy()

        tr_acc = tr.copy()
        tr_acc.differentiate()
        tr_acc.stats.channel='acc'

        tr_dis = tr.copy()
        tr_dis.integrate().detrend("linear")
        tr_dis.stats.channel='dis'

        # Window beyond pick to search for maximum amplitude
        max_amp_win = .02

        #for phase in ['P']:
        for phase in ['P', 'S']:

            dd = {}
            pick_time = pick_dict[sta][phase].time
            ipick = int((pick_time - tr.stats.starttime) * tr.stats.sampling_rate)

            # Put vel[ipick] = 0 to measure zero crossing and polarity
            #tr.data = tr.data - tr.data[ipick]
            (polarity, icross) = measure_velocity_polarity(tr, ipick)

            nmax_len = int(max_amp_win * tr.stats.sampling_rate)
            ipeak, peak_vel = get_peak_amp(tr, ipick, icross)
            imax, max_vel   = get_peak_amp(tr, ipick, ipick + nmax_len)

            tpeak= tr.stats.starttime + float(ipeak * tr.stats.delta)
            tmax = tr.stats.starttime + float(imax * tr.stats.delta)
            tcross = tr.stats.starttime + float(icross * tr.stats.delta)

            if debug:
                print("Trace: sta:%s cha:%s" % (sta, tr.stats.channel))
                print("[%s] Vel pol=%d tpick=%s" % \
                    (phase, polarity, pick_time))
                print("              tpeak=%s peak_vel=%12.10g" % (tpeak, peak_vel))
                print("             tcross=%s" % tcross)
                print("               tmax=%s max_vel=%12.10g" % (tmax, max_vel))

            if plot:
                #tr.plot()
                plot_channels_with_picks(Stream(traces=[tr]), sta, picks, title="sta:%s cha:%s" % (sta, tr.stats.channel))

            dd['pick_time']= pick_time
            dd['polarity'] = polarity
            dd['peak_vel'] = peak_vel
            dd['max_vel']  = max_vel
            dd['tpeak_vel'] = tpeak
            dd['tmax_vel']  = tmax
            #dd['vel_period']  = vel_period

            # Put the disp[ipick] = 0 to measure pulse width/area
            tr_dis.data = tr_dis.data - tr_dis.data[ipick]
            dis_polarity = np.sign(tr_dis.data[icross])
            pulse_width, pulse_area = measure_displacement_pulse(tr_dis, ipick, icross)

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

                if plot:
                    #tr_dis.plot()
                    plot_channels_with_picks(Stream(traces=[tr_dis]), sta, picks, title="sta:%s cha:%s" % (sta, tr.stats.channel))


            # This is likely temporary and can be removed once we are working
            # with automatic picks that already have snr set

            dd['snr'] = calculate_snr(Stream(traces=[tr]), pick_time, pre_wl=.03, post_wl=.03)
            #print("%s: snr=%f" % (tr.get_id(),snr))

            key = "%s_arrival" % phase

    # TODO Need to check so we don't overwrite P_arrival/S_arrival dict if already created
            #if key in tr.stats:
                #tr.stats[key]

            tr.stats[key] = dd

            #if plot:
                #st3 = Stream(traces = [tr_dis, tr, tr_acc])
                #plot_channels_with_picks(st3, sta, picks, title="sta:%s cha:%s" % (sta, tr.stats.channel))


    return



def measure_velocity_polarity(tr, istart, max_pulse_duration=.08):
    """
    Determine polarity and first zero crossing of pick at istart on velocity trace

    :param tr: velocity trace
    :type tr: obspy.core.trace.Trace or microquake.core.Trace
    :param istart: pick index in trace
    :type istart: int
    :param max_pulse_duration: max allowed duration (sec) beyond pick to search 
                               for first zero crossing
    :type max_pulse_duration: float

    :returns: polarity, icross: polarity = {-1,0,1}, icross=index of first zero cross in trace
    :rtype: int, int
    """

    fname = 'measure_velocity_polarity'

    # Max number of points out from pick to search for first zero crossing
    nmax = int(max_pulse_duration * tr.stats.sampling_rate)

    sign = np.sign(tr.data)

    nbuf = 6 #  Fudge factor to account for pick too early
    # Take polarity a bit beyond the pick to
    #   ensure we're truly in the arrival pulse
    polarity = sign[istart + nbuf]

    for i in range(istart + nbuf, istart + nmax):
        time = tr.stats.starttime + float(i*tr.stats.delta)
        #print("%4d %s %12.10g %d" % (i, time, tr.data[i], sign[i]))
        if sign[i] != polarity:
            break
        if i == istart + nmax - 1:
            print("%s: Unable to locate first velocity zero crossing for tr:%s!" % (fname, tr.get_id()))
            tr.plot()

    icross = i - 1

    return polarity, icross



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


