#from spp.travel_time import core
#from spp.utils import get_stations
#from microquake.core import read, UTCDateTime
#from microquake.core.stream import Stream
from obspy.core.event.base import ResourceIdentifier
#from microquake.waveform.pick import SNR_picker, calculate_snr, kurtosis_picker
#from microquake.core.event import Pick, make_pick, Arrival
import numpy as np
import os

from obspy.core.utcdatetime import UTCDateTime

from microquake.core.stream import Trace, Stream
from microquake.core.util.tools import copy_picks_to_dict
#from obspy.core.stream import Stream
#from microquake.waveform.pick import SNR_picker, calculate_snr
import matplotlib.pyplot as plt
import copy
#from lib.waveform import WaveformPlotting as wp
from mth_lib.waveform import WaveformPlotting as wp

#from spp.utils import logger as log
#logger = log.get_logger("helper", 'z.log')
#logger = log.get_logger("helper", 'helper.log')

#from liblog import getLogger
from logging import getLogger
logger = getLogger()


def get_log_level(level_number):
    log_level = {
                50: 'CRITICAL',
                40: 'ERROR',
                30: 'WARNING',
                20: 'INFO',
                10: 'DEBUG',
                 0: 'NOTSET'
                }
    if level_number in log_level.keys():
        return log_level[level_number]
    else:
        return None


def check_for_dead_trace(tr):
    eps = 1e-6
    data = tr.data.copy()
    mean = np.mean(data)
    max = np.max(data) - mean
    min = np.min(data) - mean
    #print('%s: mean:%f max:%f min:%f' % (tr.get_id(), mean, max, min))
    if max < eps and np.abs(min) < eps:
        #print('** Dead channel')
        return 1
    else:
        return 0

def arrivals_from_picks(arrivals):
    d = {}
    for arr in arrivals:
        print("arr resource_id=%s pick_id=%s" % (arr.resource_id, arr.pick_id))
        pick = arr.pick_id.get_referred_object()
        if pick is None:
            print("returned None pick!")
        else:
            print("returned pick resource_id:%s" % pick.resource_id)

        d[pick.resource_id] = arr
        print("d[pick.resource_id:%s] --> arr_resource_id:%s" % (pick.resource_id, arr.resource_id))

    return d

def get_pick_from_arrival(picks, arrival):
    for pick in picks:
        if pick.resource_id == arrival.pick_id:
            return pick
    return None

def get_arrival_from_pick(arrivals, pick):
    for arr in arrivals:
        if arr.pick_id == pick.resource_id:
            return arr
    return None

def picks_to_arrivals(picks):
    arrivals = []
    for pick in picks:
        arrival = Arrival()
        arrival.phase = pick.phase_hint
        arrival.pick_id = ResourceIdentifier(id=pick.resource_id.id, referred_object=pick)
# MTH: obspy seems to require that we do this once or else it loses the reference:
        pk = arrival.pick_id.get_referred_object()
        #print(pk)
        #arrival.pick_id = pick.resource_id.id
        arrivals.append(arrival)
    return arrivals


from spp.utils.application import Application
def plot_profile_with_picks(st, picks=None, origin=None, title=None):

    fname = 'plot_profile_with_picks'

    if origin is None:
        print('%s: need to specify origin' % fname)
        return 0

    if picks is None:
        #print(origin.time, origin.loc)
        #picks = get_predicted_picks(st, origin)
        #picks = get_predicted_picks(st.composite(), origin)

        app = Application()
        picks = app.synthetic_arrival_times(origin.loc, origin.time)

    plt.clf()

    earliest_pick = UTCDateTime(2070, 1, 1)

    for pick in picks:
        if pick.time < earliest_pick:
            earliest_pick = pick.time

    #print("plot_profile_with_picks: sort P picks")

    sorted_p_picks = sorted([pick for pick in picks if pick.phase_hint == 'P'], key=lambda x: x.time)

    pick_dict = copy_picks_to_dict(picks)

    for tr in st.composite():
        pk_p = None
        pk_s = None
        sta = tr.stats.station
        if sta in pick_dict:
            if 'P' in pick_dict[sta]:
                pk_p = pick_dict[sta]['P'].time
            if 'S' in pick_dict[sta]:
                pk_s = pick_dict[sta]['S'].time

        if pk_p is None:
            print('sta: %s pk_p is None' % (tr.get_id()))
            continue

        if pk_p:
            d = (pk_p - origin.time) * 5000
        elif pk_s:
            d = (pk_s - origin.time) * 5000/2.
        else:
            continue

        starttime = tr.stats.starttime

        rt = np.arange(0, tr.stats.npts) / tr.stats.sampling_rate
        t = np.array([starttime + dt for dt in rt])

        data = tr.data
        data /= np.max(np.abs(data))
        #print("plot_profile: tr:%s pk_p.time=%s npts=%d type=%s" % \
              #(tr.get_id(), pk_p.time, data.size, data.dtype))

        plt.plot(t, tr.data*20 + d, 'k')

        if pk_p:
            plt.vlines(pk_p, d-20, d+20, colors='r', linestyles=':')
        if pk_s:
            plt.vlines(pk_s, d-20, d+20, colors='b', linestyles='--')

    stations = st.unique_stations()
    tmax = 0.55
    for i,pick in enumerate(sorted_p_picks):
        pk_p = pick.time
        d = (pk_p - origin.time) * 5000
        station = pick.waveform_id.station_code
        #print("add sta code: %s for pk_p:%s" % (station, pk_p))
        if station not in stations:
            continue
        if i%2:
            plt.text(earliest_pick.timestamp + tmax, d, station, horizontalalignment='left', \
                     verticalalignment='center',color='red')
        else:
            plt.text(earliest_pick.timestamp - .1, d, station, horizontalalignment='right', \
                     verticalalignment='center',color='red')

    if title:
        plt.title(title, fontsize=10)
    plt.xlim(earliest_pick.timestamp - .1, earliest_pick.timestamp + tmax)
    #plt.xlim(earliest_pick.timestamp - .1, earliest_pick.timestamp + .5)
    plt.show()

def get_predicted_picks(stream, origin):

    predicted_picks = []
    for tr in stream:
        station = tr.stats.station
        ptime = core.get_travel_time_grid_point(station, origin.loc, phase='P', use_eikonal=False)
        stime = core.get_travel_time_grid_point(station, origin.loc, phase='S', use_eikonal=False)
        #ptime = core.get_travel_time_grid(station, origin.loc, phase='P', use_eikonal=False)
        #stime = core.get_travel_time_grid(station, origin.loc, phase='S', use_eikonal=False)
        predicted_picks.append( make_pick(origin.time + ptime, phase='P', wave_data=tr) )
        predicted_picks.append( make_pick(origin.time + stime, phase='S', wave_data=tr) )

        #def make_pick(time, phase='P', wave_data=None, SNR=None, mode='automatic', status='preliminary'):

    return predicted_picks


def plot_channels_with_picks(stream, station, picks, channel=None, title=None):

    fname = 'plot_channels_with_picks'

    extras = {}

    # stream.select returns a new stream with *copies* of the trace references,
    #   so any manipulation will affect the parent stream!
    #st = stream.select(station=station)
    st = stream.copy().select(station=station)

    for tr in st:
        check_for_dead_trace(tr)

    if channel is None:
        st2 = st.composite().select(station=station)
        st3 = st + st2
    else:
        st3 = st.composite().select(station=station)
        for tr in st:
            if tr.stats.channel == channel:
                st3 = st3 + Stream(traces=[tr])
                break

    if len(st3) == 0:
        print('%s: sta:%s st3 is empty --> nothing to plot!' % (fname, station))
        return

    if type(picks) is list:
        picks = copy_picks_to_dict(picks)


    pFound = sFound = 0
    if station in picks:
        if 'P' in picks[station]:
            print("plot_channel: found p_pick:%s" % picks[station]['P'].time)
            extras['ptime'] = picks[station]['P'].time
            pFound = 1
        if 'S' in picks[station]:
            extras['stime'] = picks[station]['S'].time
            sFound = 1
        if pFound:
            starttime = extras['ptime'] - .02
            starttime = extras['ptime'] - .03
            starttime = extras['ptime'] - .15
            #starttime = extras['ptime'] - .5
            endtime   = extras['ptime'] + .20
            endtime   = extras['ptime'] + .30
            endtime   = extras['ptime'] + .50
        elif sFound:
            starttime = extras['stime'] - .2
            endtime   = extras['stime'] + .2
        #st.trim(starttime, endtime, pad=True, fill_value=0.0)

        st3.trim(starttime, endtime, pad=True, fill_value=0.0)

    waveform = wp(stream=st3, color='k', xlabel='Seconds',number_of_ticks=8, tick_rotation=0, title=title, addOverlay=False, extras=extras, outfile=None)
    waveform.plot_waveform(equal_scale='False')
    return

def calc_avg_snr(stream, picks, preWl=.03, postWl=.03):
    fname = 'calc_avg_snr'

    snrs = []
    for pick in picks:
        sta = pick.waveform_id.station_code
        trs = stream.select(station=sta)
        snr = calculate_snr(trs, pick.time, preWl, postWl)
        logger.debug("%s: sta:%s phase:%s snr:%5.2f" % (fname, sta, pick.phase_hint, snr))
        snrs.append(snr)
    return np.mean(np.array(snrs))


#def check_trace_channels_with_picks(stream, picks, return_snr=False):

def remove_noisy_traces(stream, picks, pre_wl=.03, post_wl=.03, snr_thresh=1.6):

    #fname = 'check_trace_channels_with_picks'
    fname = 'remove_noisy_traces'

    noisy_chans = []

    pick_dict = copy_picks_to_dict(picks)

    for tr in stream:
        sta = tr.stats.station
        if sta in pick_dict and 'P' in pick_dict[sta]:
            p_time = pick_dict[sta]['P'].time
        else:
            print("No P pick for sta:%s --> Skip tr:%s" % (sta, tr.get_id()))
            continue

        noise_end   = p_time - .005
        noise_start = noise_end - pre_wl
        if noise_start < tr.stats.starttime:
            print("noise_start < trace starttime!")
            print("%s: sta:%s cha:%s tr_start:%s noise_start:%s P_time:%s" % \
                  (fname, sta, tr.stats.channel, tr.stats.starttime, noise_start, p_time))

        signal_start = p_time + .005
        signal_end   = signal_start + post_wl

        if signal_end >= tr.stats.endtime:
            print("signal_end >= trace endtime!")

        noise  = tr.copy().trim(noise_start, noise_end, pad=False, fill_value=0.0)
        signal = tr.copy().trim(signal_start, signal_end, pad=False, fill_value=0.0)

        snr = np.var(signal.data) / np.var(noise.data)
        #print("tr:%s snr:%f" % (tr.get_id(), snr))

        if snr < snr_thresh:
            #print("*** SNR too low --> Remove")
            #tr.plot()
            noisy_chans.append(tr)

    for tr in noisy_chans:
        #print("Remove tr:%s from st" % tr.get_id())
        stream.remove(tr)

    return noisy_chans


def calculate_residual(st, picks, origin):

    tot_resid = 0.
    rms_resid = 0.
    npicks = 0
    obs_picks = copy_picks_to_dict(picks)
    syn_picks = copy_picks_to_dict(get_predicted_picks(st, origin))

    for station in obs_picks:
        for phase in obs_picks[station]:
            resid = obs_picks[station][phase].time.timestamp - syn_picks[station][phase].time.timestamp
            tot_resid += resid
            rms_resid += resid*resid
            npicks += 1
            print('%2s: [%s] ims:%s  new:%s ims-new:%8.4f' % (station, phase, \
                obs_picks[station][phase].time, syn_picks[station][phase].time, resid))

    rms_resid /= float(npicks)
    rms_resid =  np.sqrt(rms_resid)
    print('== tot_resid=%8.4f rms_resid=%8.4f' % (tot_resid, rms_resid))

def clean_picks(st, picks, preWl=.03, postWl=.03, thresh=3):
    fname = 'clean_picks'

    pick_dict = copy_picks_to_dict(picks)

    noisy_picks = []

    for station in pick_dict:
        trs = st.select(station=station)
        if len(trs) == 0:
            print('%s: sta:%s has picks but is not in stream!' % (fname, station))
            continue
        if 'P' in pick_dict[station]:
            p_snr  = calculate_snr(trs, pick_dict[station]['P'].time, preWl, postWl)
            logger.debug('%s: sta:%s [P] snr [%.1f]' % (fname, station,p_snr))
            if p_snr < thresh:
                logger.debug('%s: sta:%s ** P snr [%.1f] is < thresh!' % (fname, station,p_snr))
                if get_log_level(logger.getEffectiveLevel()) == 'DEBUG':
                    plot_channels_with_picks(st, station, picks, title="sta:%s P snr:%.1f < thresh" % (station, p_snr))
                #del(pick_dict[station]['P'])
                noisy_picks.append(pick_dict[station]['P'])
        if 'S' in pick_dict[station]:
            s_snr  = calculate_snr(trs, pick_dict[station]['S'].time, preWl, postWl)
            logger.debug('%s: sta:%s [S] snr [%.1f]' % (fname, station,s_snr))
            snrs=[]
            for tr in trs:
                tr_snr  = calculate_snr(Stream(traces=[tr]), pick_dict[station]['S'].time, preWl, postWl)
                snrs.append((tr.stats.channel,tr_snr)) 
            logger.debug('%s: sta:%s [S] snr [%.1f] %s' % (fname, station,s_snr, snrs))

            '''
            if s_snr < thresh:
                logger.debug('%s: sta:%s ** S snr  [%.1f]is < thresh!' % (fname, station,s_snr))
                nabove=0
                snrs=[]
                for tr in trs:
                    #st2 = Stream()
                    #st2.append(tr)
                    tr_snr  = calculate_snr(Stream(traces=[tr]), pick_dict[station]['S'].time, preWl, postWl)
                    if tr_snr > thresh:
                        nabove += 1
                    #print("id:%s --> S snr:%f" % (tr.get_id(), tr_snr))
                    snrs.append(tr_snr)
                if nabove >= 2:
                    print(" ** keep it: snrs:", snrs)
                    pass
                else:
                    logger.debug(" ** remove pick")
                    #del(pick_dict[station]['S'])
                    noisy_picks.append(pick_dict[station]['S'])
                #plot_channels_with_picks(st, station, picks, title="sta:%s S snr:%.1f < thresh" % (station, s_snr))
            if station in ['67', '68', '97', '45', '31', '66', '18','67']:
            '''

            if s_snr < thresh:
                logger.debug('%s: sta:%s ** S snr  [%.1f]is < thresh!' % (fname, station,s_snr))
                if get_log_level(logger.getEffectiveLevel()) == 'DEBUG':
                    plot_channels_with_picks(st, station, picks, title="sta:%s P_snr:%.1f S_snr:%.1f" % (station, p_snr,s_snr))
                noisy_picks.append(pick_dict[station]['S'])

    return noisy_picks
    
    '''
    cleaned_picks = []
    for station in pick_dict:
        for phase in pick_dict[station]:
            cleaned_picks.append(pick_dict[station][phase])

    return cleaned_picks
    '''

    '''
    station_distance = {}
    for station in get_stations().stations():
        dx = station.loc[0] - location[0]
        dy = station.loc[1] - location[1]
        dz = station.loc[2] - location[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        station_distance[station.code] = dist
        #print(station.code, station_distance)
    '''


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def clean_nans(st, threshold=.05):
    for tr in st:
        #data  = tr.data[~np.isnan(tr.data)]
        stats = tr.stats
        nans, x= nan_helper(tr.data)
        number_of_nans = tr.data[nans].size

        logger.debug("%s: %s - %s (npts:%d n_NaN:%d) dtype:%s"% \
             (stats.station, stats.starttime, stats.endtime, stats.npts, number_of_nans, tr.data.dtype))

        #tr.plot()

        if float(number_of_nans/stats.npts) > threshold:
            logger.debug("clean_nans: Drop trace:%s --> percent_nans=%.1f" % \
                        (tr.get_id(), float(number_of_nans/stats.npts)))
            st.remove(tr)
            #continue

        else:
            tr.data[nans]= np.interp(x(nans), x(~nans), tr.data[~nans])

    return
