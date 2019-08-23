import os

import numpy as np
from loguru import logger
from numpy.linalg import norm
from obspy.core import UTCDateTime
from obspy.core.event import WaveformStreamID
from scipy.interpolate import interp1d

from microquake.core import Trace
from microquake.core.data.grid import read_grid
from microquake.core.event import Arrival, Pick
from microquake.core.helpers.velocity import get_current_velocity_model_id
from microquake.core.settings import settings
from microquake.core.simul.eik import ray_tracer


def get_grid(station_code, phase, type='time'):
    """
    get a travel time grid for a given station and a given phase
    :param station_code: station code
    :param phase: Phase ('P' or 'S')
    :param type: type of grid ('time', 'take_off', 'azimuth')
    :return:
    """
    nll_dir = settings.nll_base
    f_tt = os.path.join(nll_dir, 'time', 'OT.%s.%s.%s.buf'
                        % (phase.upper(), station_code, type))
    tt_grid = read_grid(f_tt, format='NLLOC')

    return tt_grid


def get_grid_point(station_code, phase, location,
                   grid_coordinates=False, type='time'):
    """
    get value on a grid at a given point inside the grid
    :param station_code: Station code
    :param phase: Phase ('P' or 'S')
    :param location: point where the value is interpolated
    :param grid_coordinates: whether the location is expressed in grid
    coordinates or in model coordinates (default True)
    :param type: type of grid ('time', 'take_off', 'azimuth')
    :return:
    """

    tt = get_grid(station_code, phase, type=type)

    return tt.interpolate(location, grid_coordinate=grid_coordinates)[0]


def get_ray(station_code, phase, location, grid_coordinate=False):
    """
    return a ray for a given location - station pair for a given phase
    :param station_code: station code
    :param phase: phase ('P', 'S')
    :param location: start of the ray
    :param grid_coordinate: whether start is expressed in  grid
    coordinates or model coordinates (default False)
    :return:
    """
    travel_time = get_grid(station_code, phase, type='time')

    return ray_tracer(travel_time, location,
                      grid_coordinates=grid_coordinate)


def create_arrivals_from_picks(picks, event_location, origin_time):
    """
    create a set of arrivals from a list of picks
    :param picks: list of microquake.core.event.Pick
    :param event_location: event location list, tuple or numpy array
    :param origin_time: event origin_time
    :return: list of microquake.core.event.Arrival
    """

    # print("create_arrival_from_picks: event_location:<%.1f, %.1f, %.1f>" % \
    # (event_location[0], event_location[1], event_location[2]))

    arrivals = []

    for pick in picks:
        station_code = pick.waveform_id.station_code

        arrival = Arrival()
        arrival.phase = pick.phase_hint
        phase = pick.phase_hint

        ray = get_ray(station_code, phase, event_location)
        arrival.distance = ray.length()

        # TODO: MTH: Gotta think about how to store the ray points. Obspy will not handle
        #       a list in the extra dict, so you won't be able to do something like event.copy() later
        # arrival.ray = list(ray.nodes)
        # for node in ray.nodes:
        # print(node)

        # xoff = ray.nodes[-2][0] - ray.nodes[-1][0]
        # yoff = ray.nodes[-2][1] - ray.nodes[-1][1]
        # zoff = ray.nodes[-2][2] - ray.nodes[-1][2]
        # baz = np.arctan2(xoff,yoff)
        # if baz < 0:
        # baz += 2.*np.pi

        # pick.backazimuth = baz*180./np.pi

        predicted_tt = get_grid_point(station_code, phase,
                                      event_location)
        predicted_at = origin_time + predicted_tt
        arrival.time_residual = pick.time - predicted_at
        # print("create_arrivals: sta:%3s pha:%s pick.time:%s

        arrival.takeoff_angle = get_grid_point(station_code, phase,
                                               event_location, type='take_off')
        arrival.azimuth = get_grid_point(station_code, phase,
                                         event_location, type='azimuth')
        # print("create arrival: type(arrival)=%s type(takeoff_angle)=%s type(azimuth)=%s" % \
        # (type(arrival), type(arrival.takeoff_angle), type(arrival.azimuth)))

        # MTH: arrival azimuth/takeoff should be in degrees - I'm pretty sure the grids
        #  store them in radians (?)
        arrival.azimuth *= 180./np.pi

        if arrival.azimuth < 0:
            arrival.azimuth += 360.
        arrival.takeoff_angle *= 180./np.pi

        arrival.pick_id = pick.resource_id.id
        arrival.earth_model_id = get_current_velocity_model_id(phase)
        arrivals.append(arrival)

    return arrivals


def estimate_origin_time(stream, event_location):
    """
    estimate the origin time given an estimate of the event location and
    a set of traces
    :param stream: a microquake.core.Stream object containing a series
    of traces
    :param event_location: event location (list, tuple or numpy array)
    :return: estimate of the origin time
    """
    # import matplotlib.pyplot as plt

    start_times = []
    end_times = []
    sampling_rates = []
    stream = stream.detrend('demean')

    for trace in stream:
        start_times.append(trace.stats.starttime.datetime)
        end_times.append(trace.stats.endtime.datetime)
        sampling_rates.append(trace.stats.sampling_rate)

    min_starttime = UTCDateTime(np.min(start_times)) - 1.0
    max_endtime = UTCDateTime(np.max(end_times))
    max_sampling_rate = np.max(sampling_rates)

    shifted_traces = []
    npts = np.int((max_endtime - min_starttime) * max_sampling_rate)
    t_i = np.arange(0, npts) / max_sampling_rate

    for phase in ['P', 'S']:
        for trace in stream.composite():
            station = trace.stats.station
            tt = get_grid_point(station, phase, event_location)
            trace.stats.starttime = trace.stats.starttime - tt
            data = np.nan_to_num(trace.data)

            # dividing by the signal std yield stronger signal then
            # dividing by the max. Dividing by the max amplifies the
            # noisy traces as signal is more homogeneous on these traces
            data /= np.std(data)
            # data /= np.max(np.abs(data))
            sr = trace.stats.sampling_rate
            startsamp = int((trace.stats.starttime - min_starttime) *
                            trace.stats.sampling_rate)
            endsamp = startsamp + trace.stats.npts
            t = np.arange(startsamp, endsamp) / sr
            try:
                f = interp1d(t, data, bounds_error=False, fill_value=0)
            except:
                continue

            shifted_traces.append(np.nan_to_num(f(t_i)))

    shifted_traces = np.array(shifted_traces)

    w_len_sec = 50e-3
    w_len_samp = int(w_len_sec * max_sampling_rate)

    stacked_trace = np.sum(np.array(shifted_traces) ** 2, axis=0)
    stacked_trace /= np.max(np.abs(stacked_trace))
    #
    i_max = np.argmax(np.sum(np.array(shifted_traces) ** 2, axis=0))

    if i_max - w_len_samp < 0:
        pass

    stacked_tr = Trace()
    stacked_tr.data = stacked_trace
    stacked_tr.stats.starttime = min_starttime
    stacked_tr.stats.sampling_rate = max_sampling_rate

    o_i = np.argmax(stacked_tr)
    # k = kurtosis(stacked_tr, win=30e-3)
    # diff_k = np.diff(k)

    # o_i = np.argmax(np.abs(diff_k[i_max - w_len_samp: i_max + w_len_samp])) + \
    #       i_max - w_len_samp

    origin_time = min_starttime + o_i / max_sampling_rate
    # Tracer()()

    return origin_time


def fix_arr_takeoff_and_azimuth(cat, vp_grid, vs_grid):
    """
    Currently NLLoc is *not* calculating the takeoff angles at the source.
    These default to -1 so that when microquake.nlloc reads last.hyp it
    returns -1 for these values.

    Here we re-create the arrivals from the picks & the NLLoc location
    so that it populates the takeoff and azimuth angles.
    Also, we add the relevant angles at the receiver (backazimuth and incidence)
    to the arrivals.
    """

    for event in cat:
        origin = event.preferred_origin()
        ev_loc = origin.loc

        vp = vp_grid.interpolate(ev_loc)[0]
        vs = vs_grid.interpolate(ev_loc)[0]

        picks = []

        for arr in origin.arrivals:
            picks.append(arr.pick_id.get_referred_object())

# MTH: create_arrivals_from_picks will create an entirely new set of arrivals (new resource_ids)
#      it will set arr.distance (looks exactly same as nlloc's arr.distance)
#      it will set arr.time_residual *** DIFFERS *** from arr.time_residual nlloc calcs/reads from last.hypo
#      it will fix the missing azim/theta that nlloc set to -1
#      it will drop nlloc arr.time_weight field

        arrivals = create_arrivals_from_picks(picks, ev_loc, origin.time)

# Now set the receiver angles (backazimuth and incidence angle)

        for arr in arrivals:
            pk = arr.pick_id.get_referred_object()
            sta = pk.waveform_id.station_code
            pha = arr.phase

            st_loc = settings.inventory.get_station(sta).loc

            xoff = ev_loc[0]-st_loc[0]
            yoff = ev_loc[1]-st_loc[1]
            zoff = np.abs(ev_loc[2]-st_loc[2])
            H = np.sqrt(xoff*xoff + yoff*yoff)
            alpha = np.arctan2(zoff, H)
            beta = np.pi/2. - alpha
            takeoff_straight = alpha * 180./np.pi + 90.
            inc_straight = beta * 180./np.pi

            if pha == 'P':
                v = vp
                v_grid = vp_grid
            elif pha == 'S':
                v = vs
                v_grid = vs_grid

            p = np.sin(arr.takeoff_angle*np.pi/180.) / v

            v_sta = v_grid.interpolate(st_loc)[0]

            inc_p = np.arcsin(p*v_sta) * 180./np.pi

            # I have the incidence angle now, need backazimuth so rotate to P,SV,SH
            back_azimuth = np.arctan2(xoff, yoff) * 180./np.pi

            if back_azimuth < 0:
                back_azimuth += 360.

            arr.backazimuth = back_azimuth
            arr.inc_angle = inc_p

            '''
            print("%3s: [%s] takeoff:%6.2f [stx=%6.2f] inc_p:%.2f [inc_stx:%.2f] baz:%.1f [az:%.1f]" % \
                (sta, arr.phase, arr.takeoff_angle, takeoff_straight, \
                inc_p, inc_straight, back_azimuth, arr.azimuth))
            '''

        origin.arrivals = arrivals

    return


def synthetic_arrival_times(event_location, origin_time):
    """
    calculate synthetic arrival time for all the station and returns a
    list of microquake.core.event.Pick object
    :param event_location: event location
    :param origin_time: event origin time
    :return: list of microquake.core.event.Pick
    """

    picks = []

    stations = settings.inventory.stations()

    for phase in ['P', 'S']:
        for station in stations:
            # station = station.code
            # st_loc = site.select(station=station).stations()[0].loc

            st_loc = station.loc

            dist = norm(st_loc - event_location)

            if (phase == 'S') and (dist < 100):
                continue

            try:
                at = origin_time + get_grid_point(station.code, phase,
                                                  event_location,
                                                  grid_coordinates=False)
            # Catching error when grid file do not exist
            except FileNotFoundError as exc:
                logger.warning(
                    f'Cannot read grid for station {station.code}'
                    f' ({station.site.name}), phase {phase}: {exc}')
                continue

            wf_id = WaveformStreamID(
                network_code=settings.get('project_code'),
                station_code=station.code)
            # station_code=station)
            pk = Pick(time=at, method='predicted', phase_hint=phase,
                      evaluation_mode='automatic',
                      evaluation_status='preliminary', waveform_id=wf_id)

            picks.append(pk)

    return picks
