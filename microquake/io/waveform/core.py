# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
# Purpose: plugin for reading and writing various waveform format expending
# the number of format readable.
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
plugin for reading and writing various waveform format expending
# the number of format readable.

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from microquake.core import logger
from microquake.core.util.decorator import uncompress_file as uncompress
import logging
from struct import unpack
from datetime import datetime
from io import BytesIO
import numpy as np


def mseed_date_from_header(block4096):

    vals = unpack('>HHBBBBH', block4096[20:30])
    year, julday, hour, minute, sec, _, sec_frac = vals
    tstamp = '%0.4d,%0.3d,%0.2d:%0.2d:%0.2d.%0.4d' % (year, julday, hour, minute, sec, sec_frac)
    dt = datetime.strptime(tstamp, '%Y,%j,%H:%M:%S.%f')
    return dt


# def mseed_decomposer(stream):
#     """
#     Create an mseed files and breaks it 512 bytes mseeds
#     :param stream: stream data
#     :type stream: microquake.core.stream.Stream
#     :return: Dictionary containing a list of keys and the mseed file chunks
#     """

#     obj = BytesIO()
#     stream.write(obj, format='MSEED')

#     mseed_byte_array = obj.getvalue()

#     mseed_chunk_size = 4096
#     keys = []
#     blobs = []

#     starts = arange(0, len(mseed_byte_array), mseed_chunk_size)

#     for start in starts:
#         end = start + mseed_chunk_size
#         chunk = mseed_byte_array[start:end]

#         y = unpack('>H',chunk[20:22])[0]
#         DoY = unpack('>H', chunk[22:24])[0]
#         H = unpack('>B', chunk[24:25])[0]
#         M = unpack('>B', chunk[25:26])[0]
#         S = unpack('>B', chunk[26:27])[0]
#         r = unpack('>B', chunk[27:28])[0]
#         ToMS = unpack('>H', chunk[28:30])[0]

#         dt = datetime.strptime('%s/%0.3d %0.2d:%0.2d:%0.2d.%0.3d'
#                                % (y, DoY, H, M, S, ToMS),
#                                '%Y/%j %H:%M:%S.%f')
#         keys.append(dt)
#         blobs.append(chunk)

#     return {'key': keys, 'blob': blobs}


def read_IMS_ASCII(path, net='', **kwargs):
    """
    read a IMS_ASCII seismogram from a single station
    :param path: path to file
    :return: microquake.core.Stream
    """

    from microquake.core import Stream, Trace, Stats, UTCDateTime
    from datetime import datetime, timedelta
    import numpy as np

    data = np.loadtxt(path, delimiter=',', skiprows=1)
    stats = Stats()

    header = {}

    with open(path) as fid:
        field = fid.readline().split(',')

    stats.sampling_rate = float(field[1])
    timetmp = datetime.fromtimestamp(float(field[5])) \
      + timedelta(seconds=float(field[6]) / 1e6)  # trigger time in second

    trgtime_UTC = UTCDateTime(timetmp)
    stats.starttime = trgtime_UTC - float(field[10]) / stats.sampling_rate
    stats.npts = len(data)

    stats.station = field[8]
    stats.network = net

    traces = []
    component = np.array(['X', 'Y', 'Z'])
    std = np.std(data, axis=0)
    mstd = np.max(std)
    for k, dt in enumerate(data.T):
        stats.channel = '%s' % (component[k])
        traces.append(Trace(data=np.array(dt), header=stats))

    # Trimming the seismogram so the length does not exceed maxSeisLenght

    return Stream(traces=traces)


def read_IMS_CONTINUOUS(dname, site=None, site_list=None, event_name=None, **kwargs):
    """
    read continuous IMS ASCII data and turn them into a valid stream with
    network, station and component information properly filled
    :param dname: directory where the files for all site are stored
    :param site: a site object containing sensor information
    :type site: ~microquake.core.station.Site
    :param site_list: list of site number to read
    :type site_list: list
    :param event_name: root of the event name, to select a particular event should there be many event in a single
    directory
    :type event_name: basestring
    :return: ~microquake.core.stream.Stream
    """

    from microquake.core import Stream, Trace, Stats
    import numpy as np
    from glob import glob
    from pytz import timezone
    from datetime import datetime, timedelta
    from microquake.core import UTCDateTime

    #if not site:
    #    logger.warning('A site object should be provided for the information '
    #                   'on the system to be properly added to the traces '
    #                   'header. The file will be read but information such as '
    #                   'the site, network, station or component information '
    #                   'will not be appended')



    traces = []
    for directory in glob(dname +'/site*'):
        if site_list:
            site_no = int(directory.split('site')[-1])
            if site_no not in site_list:
                continue
        print(directory)
        # reading data
        # racer()()
        if event_name:
            sfile = glob(directory + '\%s.casc' % event_name)
        else:
            sfile = glob(directory + '\*.casc')

        if not sfile:
            continue
        sfile = sfile[0]
        stats = Stats()
        with open(sfile) as datafile:
            data = []
            X = []
            Y = []
            Z = []
            for i, line in enumerate(datafile):
                # reading header
                field = line.split(',')
                if i == 0:
                    timetmp = datetime.fromtimestamp(float(field[5]),
                                                 tz=timezone('UTC')) \
                    + timedelta(seconds=float(field[6]) / 1e6)

                    stats.starttime = UTCDateTime(timetmp)
                    stats.sampling_rate = float(field[2])
                    stats.station = field[8]

                else:
                    try:
                        X.append(float(field[2]))
                    except:
                        pass

                    try:
                        Y.append(float(field[3]))
                    except:
                        pass

                    Z.append(float(field[4]))


            stats.npts = len(Z)
            if X:
                stats.channel = 'X'
                traces.append(Trace(data=np.array(X), header=stats))
                stats.channel = 'Y'
                traces.append(Trace(data=np.array(Y), header=stats))

            stats.channel = 'Z'
            traces.append(Trace(data=np.array(Z), header=stats))

    return Stream(traces=traces)


@uncompress
def read_ESG_SEGY(fname, site=None, **kwargs):
    """
    read data produced by ESG and turn them into a valid stream with network,
    station and component information properly filled
    :param fname: the filename
    :param site: a site object containing sensor information
    :type site: ~microquake.core.station.Site
    :return: ~microquake.core.stream.Stream
    """

    from microquake.core import read, Stream, Trace
    import numpy as np

    if not site:
        logger.warning('A site object should be provided for the information '
                       'on the system to be properly added to the traces '
                       'header. The file will be read but information such as '
                       'the site, network, station or component information '
                       'will not be appended')

        return read(fname, format='SEGY',  unpack_trace_headers=True)

    st = read(fname, format='SEGY',  unpack_trace_headers=True)

    stations = site.stations()
    traces = []
    for k, tr in enumerate(st):
        x = tr.stats.segy.trace_header.group_coordinate_x
        y = tr.stats.segy.trace_header.group_coordinate_y

        hdist_min = 1e10
        station = None
        network = None
        for net in site:
            for sta in net:
                hdist = np.linalg.norm([sta.x - x, sta.y -y])
                if hdist < hdist_min:
                    hdist_min = hdist
                    station = sta
                    network = net

        tr.stats.station = station.code
        tr.stats.network = network.code
        traces.append(Trace(trace=tr))

    st2 = Stream(traces=traces)
    traces = []
    for station in site.stations():
        if np.all(station.loc == [1000,1000,1000]):
            continue
        sttmp = st2.copy().select(station=station.code)
        for k, tr in enumerate(sttmp):
            if k == 0:
                if len(sttmp) == 3:
                    tr.stats.channel = 'X'
                else:
                    tr.stats.channel = 'Z'
            elif k == 1:
                tr.stats.channel = 'Y'
            else:
                tr.stats.channel = 'Z'

            msec_starttime = tr.stats.segy.trace_header.lag_time_A
            usec_starttime = tr.stats.segy.trace_header.lag_time_B

            usecs = msec_starttime / 1000. + usec_starttime / 1.0e6
            tr.stats.starttime = tr.stats.starttime + usecs

            traces.append(Trace(trace=tr))

    return Stream(traces=traces)


@uncompress
def isTDMS(filename):
    """
    Checks whether or not the given file is a TDMS file.

    :type filename: str
    :param filename: TDMS file to be checked.
    :rtype: bool
    :return: ``True`` if a TDMS file.
    """

    try:
        tdms = TdmsFile(filename)
    except:
        return False

    return True


@uncompress
def readTDMS(filename, **kwargs):
    """
    Reads a National Instrument TDMS file and returns an microquake Stream
    object.

    .. warning::
        This function should NOT be called directly, it registers via the
        microquake :func:`~microquake.core.stream.read` function, call this
        instead.

    :type filename: str
    :param filename: SEG Y rev1 file to be read.
    :returns: A uquakepy :class:`~uquakepy.core.stream.Stream` object.

    .. rubric:: Example

    >>> from uquakepy import read
    >>> st = read("/path/to/00001034.sgy_first_trace")
    >>> st  # doctest: +ELLIPSIS
    <uquakepy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    """

    try:
        tdms = TdmsFile(filename)
    except:
        print('Not able to read %s' % filename)
        print('exiting')
        return

    traces = []
    for group in tdms.groups():
        time = tdms.object(group,'Timestamps').data
        sr = np.round(1/((time[-1]-time[0]).total_seconds() \
                         /float(len(time))))
        sr = int(sr)

        for key in tdms.objects.keys():
            if key.split('/')[-1][1:-1] == group:
                continue

            if group in key:
                channel = key.split('/')[-1][1:-1]
                if 'Timestamps' in channel:
                    continue
                elif 'Sensor' in channel:
                    channel_no = int(channel.split(' ')[-1])

                    data = tdms.object(group,channel).data

                    tmp = Trace(data = data)
                    tmp.stats.network = network
                    tmp.stats.sampling_rate = int(sr)
                    tmp.stats.location = '00'
                    starttime = UTCDateTime(time[0].replace(tzinfo = None))
                    tmp.stats.starttime = starttime
                    channel = SensorInfo['Component'][i]
                    tmp.stats.channel = channel
                    tmp.stats.station = channel  # SensorInfo['SensorID'][i]
                    traces.append(tmp)

    return Stream(traces=traces)


@uncompress
def read_hsf(filename, **kwargs):
    """
    Reads a National Instrument TDMS file and returns an microquake Stream
    object.

    .. warning::
        This function should NOT be called directly, it registers via the
        microquake :func:`~microquake.core.stream.read` function, call this
        instead.

    :param filename: filename
    :param kwargs:
    :return: ~microquake.core.stream.Stream
    """

    # TODO this functio is still being developped. Real name of sensor should
    #  be found in the file

    from struct import unpack
    import numpy as np

    with open(filename, 'rb') as hsffile:

        # reading header information
        hsffile.seek(42)
        (sampling_rate,) = unpack('i', hsffile.read(4))
        hsffile.seek(68)
        (npoints,) = unpack('i', hsffile.read(4))
        hsffile.seek(76)
        (nstations,) = unpack('i', hsffile.read(4))
        hsffile.seek(80)
        (nsensors,) = unpack('i', hsffile.read(4))
        hsffile.seek(84)
        (nchannels,) = unpack('i', hsffile.read(4))
        # hsffile.seek()
        hsffile.seek(216)
        #site_name = repr(hsffile.read(25))
        site_name = unpack('25s', hsffile.read(25))

        # reading event information
        hsffile.seek(108)
        # (unk,) = unpack('i', hsffile.read(4))
        (ex,) = unpack('f', hsffile.read(4))
        (ey,) = unpack('f', hsffile.read(4))
        (ez,) = unpack('f', hsffile.read(4))

        # reading station block
        station_block_start = 309
        station_block_size = 26
        hsffile.seek(station_block_start)
        (unk,) = unpack('f', hsffile.read(4))
        # there is likely other valuable information beside the station name
        stname = []
        for i in range(0, nstations):
            hsffile.seek(station_block_start + station_block_size * i)
            (tmp,) = unpack('14s', hsffile.read(14))
            tmp = tmp[:tmp.find('\x00')]
            stname.append(tmp)

        # reading sensor block (arrival time must be in this section)
        sensor_block_start = station_block_start + station_block_size * \
                                                   (nstations + 1)
        sensor_block_size = 222
        sensor_name_size = 14  # this is the minimum
        sensor_name_offset = 96

        senname = []
        for i in range(0, nsensors):
            hsffile.seek(sensor_block_start + sensor_block_size * i +
                         sensor_name_offset)
            (tmp,) = unpack('14s', hsffile.read(sensor_name_size))
            tmp = tmp[:tmp.find('\x00')]
            senname.append(tmp)

        # reading channel block size
        channel_block_start = sensor_block_start + sensor_block_size * \
                                                   (nsensors)
        channel_block_size = 158
        channel_name_offset = 43
        channel_name_size = 20

        chnames = []
        for i in range(0, nchannels):
            hsffile.seek(channel_block_start + channel_block_size * i +
                         channel_name_offset)
            (tmp,) = unpack('%ds' % channel_name_size, hsffile.read(
                channel_name_size))
            tmp = tmp[:tmp.find('\x00')]
            chnames.append(tmp)

        # hsffile.seek(channel_block_start)
        # for i in range(0, nchannel):

        hsffile.seek(0, 2)

        hsffile.seek(-npoints * nchannels * 4 - 2570, 2)
        npts = nchannels * npoints
        # data = [unpack('i', hsffile.read(4)) for i in range(0, npts)]
        data = np.array(unpack('i' * npts, hsffile.read(4 * npts)))
        data = data.reshape(npoints, nchannels)

        from microquake.core.stream import Trace, Stream
        trs = []
        name_ct = 0
        for k in range(0, nchannels):
            chname = chnames[k]
            tr = Trace(data=data[:,k])
            # TODO does not work quite well. Will need to collect more
            # accurate information on the associate of channel and station
            name_ct += 1
            if '1' in  chname.split('_')[-1]:
                tr.stats.channel = 'X'
            elif '2' in chname.split('_')[-1]:
                tr.stats.channel = 'Y'
            else:
                tr.stats.channel = 'Z'
            tr.stats.station = str(name_ct)
            tr.stats.sampling_rate = sampling_rate
            tr.network = 'net' # TODO change that
            # TODO find the start time
            # tr.stats.starttime =
            trs.append(tr)

        st = Stream(traces = trs)


        # TODO reading the picks and few other quantity

        # scanning file for number of channel

        # Tracer()()
        hsffile.seek(0)
        # for i in range(0, 300, 4):
        #    hsffile.seek(i)
        #    (nstation,) = unpack('d', hsffile.read(8))
        #    if '1204' in str(nstation):
        #        print 'youppi', i, nstation


@uncompress
def read_TEXCEL_CSV(filename, **kwargs):
    """
    Reads a texcel csv file and returns a microquake Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        microquake :func:`~microquake.core.stream.read` function, call this
        instead.
    :param filename: the path to the file
    :param kwargs:
    :return: ~microquake.core.stream.Stream
    """

    from microquake.core import Stream, Trace
    from microquake.core.trace import Stats
    from dateutil.parser import parse
    from datetime import timedelta
    import numpy as np
    from microquake.core import UTCDateTime

    with open(filename) as fle:
        x = []
        y = []
        z = []
        for k, line in enumerate(fle):
            if k == 0:
                if 'MICROPHONE' in line:
                    offset = 9
                else:
                    offset = 8
            # header
            if k < 2:
                continue

            val = line.strip().split(',')

            # relative time
            if k == 3:
                rt0 = timedelta(seconds=float(val[0]))

            elif k == 6:
                station = str(eval(val[offset]))

            elif k == 7:
                date = val[offset]

            elif k == 8:
                date_time = date + " " + val[offset]
                datetime = parse(date_time)
                starttime = datetime + rt0

            elif k == 9:
                site = val[offset]

            elif k == 10:
                location = val[offset]


            elif k == 17:

                sensitivity_x = float(val[offset])
                sensitivity_y = float(val[offset + 1])
                sensitivity_z = float(val[offset + 2])

            elif k == 18:
                range_x = float(val[offset])
                range_y = float(val[offset + 1])
                range_z = float(val[offset + 2])

            elif k == 19:
                trigger_x = float(val[offset])
                trigger_y = float(val[offset + 1])
                trigger_z = float(val[offset +2])

            elif k == 20:
                si_x = float(val[offset])
                si_y = float(val[offset + 1])
                si_z = float(val[offset + 2])

            elif k == 21:
                sr_x = float(val[offset])
                sr_y = float(val[offset + 1])
                sr_z = float(val[offset + 2])

            x.append(float(val[1]))
            y.append(float(val[2]))
            z.append(float(val[3]))

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        stats = Stats()
        stats.network = site
        stats.delta = si_x / 1000.0
        stats.npts = len(x)
        stats.location = location
        stats.station = station
        stats.starttime = UTCDateTime(starttime)

        stats.channel = 'radial'
        tr_x = Trace(data=x / 1000.0, header=stats)

        stats.delta = si_y / 1000.0
        stats.channel = 'transverse'
        tr_y = Trace(data=y / 1000.0, header=stats)

        stats.delta = si_z / 1000.0
        stats.channel = 'vertical'
        tr_z = Trace(data=z / 1000.0, header=stats)

    return Stream(traces=[tr_x, tr_y, tr_z])

