# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: rest_api.py
#  Purpose: module to interact with the IMS RESTAPI
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
module to interact IMS web API

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from logging import getLogger, INFO


def get_continuous(base_url, start_datetime, end_datetime,
                   site_ids, format='binary-gz', network='',
                   sampling_rate=6000., nan_limit=10, logger_level=INFO):
    """
    :param base_url: base url of the IMS server
    example: http://10.95.74.35:8002/ims-database-server/databases/mgl
    :param start_datetime: request start time (if not localized, UTC assumed)
    :type start_datetime: datetime.datetime
    :param end_datetime: request end time (if not localized, UTC assumed)
    :type end_datetime: datetime.datetime
    :param site_ids: list of sites for which data should be read
    :type site_ids: list or integer
    :param format: Requested data format ('possible values: binary and binary-gz')
    :type format: str
    :param network: Network name (default = '')
    :type network: str
    :param dtype: output type for mseed
    :return: microquake.core.stream.Stream
    """

    """
    binary file structure:
    * a binary header of size N bytes, consisting of 
        - header size written as int32
        - netid written as int32
        - siteid written as int32
        - start time written as int64(time in nanoseconds)
        - end time written as int64(time in nanoseconds)
        - netADC id written as int32
        - sensor id written as int32
        - attenuator id written as int32
        - attenuator configuration id written as int32
        - remainder of bytes(N minus total so far) written as zero
        padded.
    * A sequence of 20 - byte samples, each consisting of 
        - sample timestamp, written as int64(time in nanoseconds)
        - raw X value as float32
        - raw Y value as float32
        - raw Z value as float32
    """

    import requests
    from gzip import GzipFile
    import struct
    import numpy as np
    from microquake.core import Trace, Stream, UTCDateTime
    import sys
    from time import time as timer

    logger = getLogger('microquake.IMS.web_api')
    logger.level = logger_level

    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO, BytesIO

    if isinstance(site_ids, int):
        site_ids = [site_ids]

    start_datetime_utc = UTCDateTime(start_datetime)
    end_datetime_utc = UTCDateTime(end_datetime)
    reqtime_start_nano = int(start_datetime_utc.timestamp * 1e6) * int(1e3)
    reqtime_end_nano = int(end_datetime_utc.timestamp * 1e6) * int(1e3)
    url_cont = base_url + '/continuous-seismogram?' + \
               'startTimeNanos=%d&endTimeNanos=%d&siteId' + \
               '=%d&format=%s'

    stream = Stream()
    for site in site_ids:
        ts_processing = timer()

        if type(site) == str:
            site = int(site)

        url = url_cont % (reqtime_start_nano, reqtime_end_nano, site, format)
        url = url.replace('//', '/').replace('http:/', 'http://')

        logger.info("Getting trace for station %d\nstarttime: %s\n"
                    "endtime:   %s" % (site, start_datetime, end_datetime))

        ts = timer()
        r = requests.get(url, stream=True)

        if r.status_code != 200:
            raise Exception('request failed! \n %s' % url)
            continue
        if format == 'binary-gz':
            fileobj = GzipFile(fileobj=BytesIO(r.content))
        elif format == 'binary':
            fileobj = BytesIO(r.content)
        else:
            raise Exception('unsuported format!')
            continue

        fileobj.seek(0)
        te = timer()
        logger.info('Completing request in %f seconds' % (te - ts))

        # Reading header
        # try:
        if len(r.content) < 44:
            continue
        ts = timer()
        header_size = struct.unpack('>i', fileobj.read(4) )[0]
        net_id = struct.unpack('>i', fileobj.read(4))[0]
        site_id = struct.unpack('>i', fileobj.read(4))[0]
        starttime = struct.unpack('>q', fileobj.read(8))[0]
        endtime = struct.unpack('>q', fileobj.read(8))[0]
        netADC_id = struct.unpack('>i', fileobj.read(4))[0]
        sensor_id = struct.unpack('>i', fileobj.read(4))[0]
        attenuator_id = struct.unpack('>i', fileobj.read(4))[0]
        attenuator_config_id = struct.unpack('>i', fileobj.read(4))[0]
        te = timer()
        logger.info('Unpacking header in %f seconds' % (te - ts))

        ts = timer()
        # Reading data
        fileobj.seek(header_size)
        content = fileobj.read()

        time, sigs = strided_read(content)

        time_norm = (time - time[0]) / 1e9
        nan_ranges = get_nan_ranges(time_norm, sampling_rate, limit=nan_limit)

        tstart_norm_new = (reqtime_start_nano - time[0]) / 1e9
        tend_norm_new = (reqtime_end_nano - time[0]) / 1e9
        time_new = np.arange(tstart_norm_new, tend_norm_new, 1. / sampling_rate)

        newsigs = np.zeros((len(sigs), len(time_new)), dtype=np.float32)
        for i in range(len(sigs)):
            newsigs[i] = np.interp(time_new, time_norm, sigs[i])

        nan_ranges_ix = ((nan_ranges - time_new[0]) * sampling_rate).astype(int)

        for chan in newsigs:
            for lims in nan_ranges_ix:
                chan[lims[0]:lims[1]] = np.nan

        te = timer()
        logger.info("Unpacking data in %f seconds for %d points"
                    % (te - ts, len(time_new)))

        chans = ['X', 'Y', 'Z']

        for i in range(len(newsigs)):
            tr = Trace(data=newsigs[i])
            tr.stats.sampling_rate = sampling_rate
            tr.stats.network = str(network)
            tr.stats.station = str(site)
            tr.stats.starttime = start_datetime_utc
            tr.stats.channel = chans[i]
            stream.append(tr)

        te_processing = timer()
        logger.info("Processing completed in %f" % (te_processing -
                                                    ts_processing))

    return stream


def get_nan_ranges(tnorm, sr, limit):
    # limit is minimum consecutive missing dt's to assign nans
    diff = np.diff(tnorm) * sr
    ibad = np.where(diff > limit)[0]
    miss_start = tnorm[ibad]
    miss_lens = diff[ibad] / sr
    nan_ranges = np.vstack((miss_start, miss_start + miss_lens)).T
    return nan_ranges


def strided_read(content):

    npts = int(len(content) / 20)
    time = np.ndarray((npts,), '>q', content, 0, (20, ))
    sigs = np.zeros((3, npts), dtype=np.float32)
    sigs[0] = np.ndarray((npts,), '>f', content, 8, (20, ))
    sigs[1] = np.ndarray((npts,), '>f', content, 12, (20, ))
    sigs[2] = np.ndarray((npts,), '>f', content, 16, (20, ))

    return time, sigs


def EpochNano2UTCDateTime(timestamp, timezone):
    """
    Convert a time stamp in nanosecond to a microquake.UTCDateTime object
    :param timezone:
    :param timestamp: timestamp expressed in nanasecond
    :return: a microquake.UTCDateTime object
    """

    from microquake.core import UTCDateTime
    from datetime import datetime

    time_local = datetime.fromtimestamp(timestamp / 1.e9)
    return UTCDateTime(time_local.replace(tzinfo=timezone))


def get_catalogue(base_url, start_datetime, end_datetime, site,
                  timezone, blast=True, event=True, accepted=True, manual=True,
                  get_arrivals=False):
    """
    read catalogue data through the REST API provided by the IMS synapse
    server and return a QuakeML object

    :param base_url: base url of the IMS server e.g.
    http://10.95.74.35:8002/ims-database-server/databases/mgl
    :param start_datetime: request start time (if not localized, UTC assumed)
    :type start_datetime: datetime.datetime
    :param end_datetime: request end time (if not localized, UTC assumed)
    :type end_datetime: datetime.datetype
    :param site: a site object containing system information
    :type site: microquake.core.data.station.Site
    :param blast: if True return blasts (default True)
    :type blast: bool
    :param event: if True return events (default True)
    :type event: bool
    :param accepted: if True only accepted events and blasts are returned (
    default True)
    :type accepted: bool
    :param manual: if True only manually processed event are returned (
    default True)
    :param get_arrivals: if True picks are also returned along with the
    catalogue.
    :return: a catalogue containing a list of events
    :rtype: microquake.core.Catalog
    """

    import calendar
    from microquake.core import UTCDateTime
    from microquake.core.event import Catalog, Event, Origin, Magnitude, \
        OriginUncertainty, ConfidenceEllipsoid
    import requests
    import pandas as pd
    import sys
    from datetime import datetime

    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    start_datetime_utc = UTCDateTime(start_datetime)
    end_datetime_utc = UTCDateTime(end_datetime)

    time_start = calendar.timegm(start_datetime_utc.timetuple()) * 1e9
    time_end = calendar.timegm(end_datetime_utc.timetuple()) * 1e9

    url = base_url + \
          '/events/csv?startTimeNanos=%d&endTimeNanos=%d&blast&params='  \
           % (time_start, time_end) \
            + 'ACCEPTED, ASSOC_SEISMOGRAM_NAMES, AUTO_PROCESSED, BLAST,' \
            + 'CORNER_FREQUENCY, DYNAMIC_STRESS_DROP, ENERGY, ENERGY_P,' \
            + 'ENERGY_S, EVENT_MODIFICATION_TIME, EVENT_NAME,' \
            + 'EVENT_TIME_FORMATTED, EVENT_TIME_NANOS, LOCAL_MAGNITUDE,' \
            + 'LOCATION_RESIDUAL, LOCATION_X, LOCATION_Y, LOCATION_Z,' \
            + 'MANUALLY_PROCESSED, NUM_ACCEPTED_TRIGGERS, NUM_TRIGGERS' \
            + 'POTENCY, POTENCY_P, POTENCY_S, STATIC_STRESS_DROP, TAP_TEST' \
            + 'TEST, TRIGGERED_SITES, USER_NAME'

    # will need to add tags for the error ellipsoid

    r = requests.get(url)

    enable = False
    for line in r.iter_lines():
        line = line.decode('utf-8')
        if "EVENT_NAME" in line:
            enable = True
            csv_string = str(line) + '\n'
            continue

        if not enable:
            continue

        if "#" in line:
            continue

        e_accepted = int(line.split(',')[1])
        e_blast = int(line.split(',')[4])
        e_automatic = int(line.split(',')[3])

        processor_name = line.split(',')[-1]

        if not (blast and event):
            if ((not e_blast) and (blast)) or ((e_blast) and (event)):
                continue

        if (accepted) and (not e_accepted):
            continue

        if (manual) and (e_automatic):
           continue

        csv_string += line + '\n'
        event_name = line.split(',')[0]

    df = pd.read_csv(StringIO(csv_string))

    events = []
    for row in df.iterrows():
        for k, element in enumerate(row[1]):
            if element == '-':
                row[1][k] = None

        event = Event()
        extra = row[1].to_dict()
        for key in extra.keys():
            if key not in event.extra_keys:
                continue
            event.__setattr__(key, extra[key])

        #  create the origin object
        origin = Origin()
        origin.x = row[1]['LOCATION_X']
        origin.y = row[1]['LOCATION_Y']
        origin.z = row[1]['LOCATION_Z']

        origin.time = EpochNano2UTCDateTime(int(row[1]['EVENT_TIME_NANOS']),
                                            timezone)

        if (row[1]['ACCEPTED'] == 1) and (row[1]['MANUALLY_PROCESSED'] == 1):
            origin.evaluation_status = 'reviewed'
            origin.evaluation_mode = 'manual'
        elif (row[1]['ACCEPTED'] == 0) and (row[1]['MANUALLY_PROCESSED'] == 1):
            origin.evaluation_status = 'rejected'
            origin.evaluation_mode = 'manual'
        elif (row[1]['ACCEPTED'] == 1) and (row[1]['MANUALLY_PROCESSED'] == 0):
            origin.evaluation_status = 'preliminary'
            origin.evaluation_mode = 'automatic'
        else:
            origin.evaluation_status = 'rejected'
            origin.evaluation_mode = 'manual'

        o_u = OriginUncertainty()
        o_u.confidence_ellipsoid = ConfidenceEllipsoid()
        origin.origin_uncertainty = o_u

        # create the magnitude object
        magnitude = Magnitude()
        magnitude.mag = -999
        magnitude.error = -999
        if row[1]['LOCAL_MAGNITUDE']:
            magnitude.mag = row[1]['LOCAL_MAGNITUDE']

        magnitude.magnitude_type = 'Mw'
        magnitude.origin_id = origin.resource_id.id

        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id.id

        event.magnitudes.append(magnitude)
        event.preferred_magnitude_id = magnitude.resource_id.id

        if row[1]['BLAST'] == 1:
            event.event_type = "explosion"
        else:
            event.event_type = "earthquake"

        event_name = row[1]['EVENT_NAME']

        if get_arrivals:
            (picks, arrivals) = get_picks(base_url, event_name, site)

            event.picks = picks
            event.preferred_origin().arrivals = arrivals

        events.append(event)

    return Catalog(events=events)


def get_seismogram(base_url, sgram_name, network_code, site_code, timezone):
    """
    Read a seismogram, one sensor (uni- or tri-axial) one event using the
    REST API interface from Synapse server and return a Stream
    :param base_url: base url of the IMS server e.g.
    http://10.95.74.35:8002/ims-database-server/databases/mgl
    :param sgram_name: Seismogram name as defined in IMS system
    :type sgram_name: string
    :param network_code: code of the network
    :type network_code: string
    :param site_code: site code
    :type site_code: str
    :return: a stream containing either 1 or 3 traces depending on the number of
    component
    :rtype: microquake.core.Stream
    """

    import requests
    from microquake.core import Trace, Stream
    from microquake.core.trace import Stats
    import numpy as np

    url = base_url + '/sgrams/assoc/read_sgram?sgramName=%s' % sgram_name

    r = requests.get(url)

    traces = []
    indata = False
    data = []
    ncomponent = 0
    for lsgram in r.iter_lines():
        lsgram = lsgram.decode('utf-8')
        if 'time-sample-0-nanos' in lsgram:
            s_starttime = EpochNano2UTCDateTime(int(lsgram.split('=')[-1]),
                                                timezone)
        if 'sampling-rate' in lsgram:
            sampling_rate = float(lsgram.split('=')[-1])
        if 'num-components' in lsgram:
            ncomponent = int(lsgram.split('=')[-1])

        if indata:
            if ncomponent == 1:
                data.append(float(lsgram.split(',')[-1]))
            if ncomponent == 3:
                tmp = [float(d) for d in lsgram.split(',')[1:]]
                data.append(tmp)

        if '#Samples' in lsgram:
            indata = True

    if ncomponent == 1:
        header = Stats()
        header.network = network_code
        header.sampling_rate = sampling_rate
        header.station = site_code
        header.channel = 'Z'
        header.starttime = s_starttime
        header.npts = len(data)
        tr = Trace(data=np.array(data).astype(np.float32), header=header)
        traces.append(tr)

    if ncomponent == 3:
        data = np.array(data).astype(np.float32)
        for k, channel in enumerate(['x', 'y', 'z']):
            header = Stats()
            header.network = network_code
            header.sampling_rate = sampling_rate
            header.station = site_code
            header.channel = channel
            header.starttime = s_starttime
            header.npts = len(data)
            tr = Trace(data=data[:, k], header=header)
            traces.append(tr)

    return Stream(traces=traces)


def get_picks(base_url, event_name, site, timezone):
    """
    Read information for one event using the REST API provided by the IMS
    synapse server and return a Catalog object.

    :param base_url: base url of the IMS server e.g.
    http://10.95.74.35:8002/ims-database-server/databases/mgl
    :param event_name: event name
    :type event_name: string
    :param site: site object containing information on the network and sensors
    :type site: microquake.core.data.Station
    :return: (list of picks, origin_time)
    :rtype: microquake.event.Catalog
    """

    import requests
    from microquake.core.event import Pick, Arrival, WaveformStreamID, Origin
    import numpy as np
    from microquake.core import UTCDateTime

    url = base_url + '/events/read_event?eventName=%s' % (event_name)
    r2 = requests.get(url)

    origin = Origin()
    picks = []
    arrivals = []
    for line in r2.iter_lines():
        line = line.decode('utf-8')
        #if 'event-time' in line:
        if 'loc-t0-nanos' in line:
            try:
                origin.time = EpochNano2UTCDateTime(int(line.split('=')[-1]),
                                                    timezone)
            except:
                origin.time = UTCDateTime.now()
        elif 'accepted' in line:
            if 'true' in line:
                origin.evaluation_status = 'reviewed'
# No information is provided to really know what the status is. Assuming manual.
                origin.evaluation_mode = 'manual'
            else:
                origin.evaluation_status = 'rejected'
                origin.evaluation_mode = 'manual'
        elif 'loc-south' in line:
            origin.y = -float(line.split('=')[-1])
        elif 'loc-west' in line:
            origin.x = -float(line.split('=')[-1])
        elif 'loc-down' in line:
            origin.z = -float(line.split('=')[-1])
#        elif 'local-magnitude' in line:


        elif 't.' in line:
            if 'index' in line:
                waveform_id = WaveformStreamID()
            elif 'site-id' in line:
                station_code = line.split('=')[-1]
                waveform_id.station_code = station_code
            elif 'accepted=false' in line:
                continue
            elif 'pick-p-time-nanos' in line:
                pick = Pick()
                arrival = Arrival()
                pick.time = EpochNano2UTCDateTime(int(line.split('=')[-1]),
                                                  timezone)
                pick.phase_hint = 'P'
                pick.waveform_id = waveform_id
                pick.evaluation_mode=origin.evaluation_mode
                pick.evaluation_status=origin.evaluation_status
                arrival.pick_id = pick.resource_id.id
                arrival.phase = 'P'

                import logging
                try:
                    station = site.select(station=station_code).stations()[0]
                except:
                    logging.warning("Station %s not found!\n The station object needs to be updated" % station_code)
                    continue

                arrival.distance = np.linalg.norm(station.loc - origin.loc)
                arrival.takeoff_angle = np.arccos((station.z - origin.z) \
                                        / arrival.distance) * 180 / np.pi
                dx = station.x - origin.x
                dy = station.y - origin.y
                arrival.azimuth = np.arctan2(dx, dy) * 180 / np.pi
                picks.append(pick)
                arrivals.append(arrival)

            elif 'pick-s-time-nanos' in line:
                pick = Pick()
                arrival = Arrival()
                pick.time = EpochNano2UTCDateTime(int(line.split('=')[-1]),
                                                  timezone)
                pick.phase_hint = 'S'
                pick.waveform_id = waveform_id
                pick.evaluation_mode=origin.evaluation_mode
                pick.evaluation_status=origin.evaluation_status
                arrival.pick_id = pick.resource_id.id
                arrival.phase = 'S'
                import logging
                try:
                    station = site.select(station=station_code).stations()[0]
                except:
                    logging.warning("Station %s not found!\n The station object needs to be updated" % station_code)
                    continue

                arrival.distance = np.linalg.norm(station.loc - origin.loc)
                arrival.takeoff_angle = np.arccos((station.z - origin.z) \
                                                  / arrival.distance) * 180 / np.pi
                dx = station.x - origin.x
                dy = station.y - origin.y
                arrival.azimuth = np.arctan2(dx, dy) * 180 / np.pi
                picks.append(pick)
                arrivals.append(arrival)

    return (picks, arrivals)


def get_picks_event(base_url, event, site):
    """
    get pick for an microquake event
    :param base_url:
    :param event:
    :param site:
    :return: event
    """

    event_name = event.EVENT_NAME

    (picks, arrivals) = get_picks(base_url, event_name, site)


    event.preferred_origin().arrivals = arrivals
    event.picks = picks

    return event


def get_seismogram_event(base_url, event, network_code):
    """
    Read the seismograms related to an event using the IMS REST API interface
    :param base_url: base url of the IMS server e.g.
    http://10.95.74.35:8002/ims-database-server/databases/mgl
    :type base_url: string
    :param event: an event containing an origins, arrivals and picks
    :type event: microquake.core.event.Event
    :param network_code:  code of the network
    :type network_code: string
    :return: a stream of traces
    :rtype: microquake.core.Stream
    """

    from microquake.core import Stream

    seismogram_names = event.ASSOC_SEISMOGRAM_NAMES.split(';')
    station_codes = event.TRIGGERED_SITES.split(';')
    traces = []
    for sname, station_code in zip(seismogram_names, station_codes):
        st = get_seismogram(base_url, sname, network_code, station_code)
        for tr in st:
            traces.append(tr)

    return Stream(traces=traces)


def get_range(base_url, start_datetime, end_datetime, site, network_code,
               blast=True, event=True, accepted=True, manual=True):

    """
    read catalogue, picks, and seismogram for a range of date through the REST
    API provided by the IMS synapse server

    :param base_url: base url of the IMS server e.g.
    http://10.95.74.35:8002/ims-database-server/databases/mgl
    :param start_datetime: request start time
    :type start_datetime: datetime.datetime
    :param end_datetime: request end time
    :type end_datetime: datetime.datetype
    :param site: a site object containing system information
    :type site: microquake.core.data.station.Site
    :param blast: if True return blasts (default True)
    :type blast: bool
    :param event: if True return events (default True)
    :type event: bool
    :param network_code: network code
    :param accepted: if True only accepted events and blasts are returned (
    default True)
    :type accepted: bool
    :param manual: if True only manually processed event are returned (
    default True)
    :param time_zone: time zone name see pytz for a list of time zones
    :return: a list of catalog and stream tuple
    """

    from microquake.core.event import Catalog

    events = get_catalogue(base_url, start_datetime, end_datetime, site,
                            blast, event, accepted, manual)

    streams = [get_seismogram_event(base_url, event, network_code) for event in \
               events]

    catalogs = [Catalog(events=[event]) for event in events]

    return [(cat, st) for cat, st in zip(catalogs, streams)]
