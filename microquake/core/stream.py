# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
#  Purpose: Expansion of the obspy.core.stream module
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.stream module

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import obspy.core.stream as obsstream
from microquake.core.trace import Trace
from microquake.core.util.decorator import uncompress_file as uncompress
import numpy as np
from microquake.core.util import ENTRY_POINTS
from microquake.core.util import tools
from pkg_resources import load_entry_point

from obspy.core.utcdatetime import UTCDateTime
from IPython.core.debugger import Tracer


class Stream(obsstream.Stream):
    __doc__ = obsstream.Stream.__doc__.replace('obspy', 'microquake')

    def __init__(self, stream=None, **kwargs):
        super(Stream, self).__init__(**kwargs)
        if stream:
            traces = []
            for tr in stream.traces:
                traces.append(Trace(trace=tr))

            self.traces = traces

    def composite_old(self):

        return composite_traces(self)

    def composite(self):
        """
        returns a new stream object containing composite trace for all station.
        The amplitude of the composite traces are the norm of the amplitude of
        the trace of all component and the phase of the trace (sign) is the sign
        of the first components of a given station.
        :param st: a stream object
        :type st: ~microquake.core.stream.Stream
        :rtype: ~microquake.core.stream.Stream

        """
        groups = self.chan_groups()
        dat, t0 = self.as_array()
        comp = tools.create_composite(dat, groups)

        stnew = Stream()
        for i, sig in enumerate(comp):
            stats = self[groups[i][0]].stats.copy()
            stats.starttime = t0
            stats.channel = 'C'
            tr = Trace(data=sig, header=stats)
            stnew.append(tr)

        return stnew

    def as_array(self, wlen_sec=None, taplen=0.05):
        t0 = np.min([tr.stats.starttime for tr in self])
        if wlen_sec is not None:
            sr = self[0].stats.sampling_rate
            npts_fix = int(wlen_sec * sr)
        else:
            npts_fix = int(np.max([len(tr.data) for tr in self]))

        return tools.stream_to_array(self, t0, npts_fix, taplen=taplen), t0

    def chan_groups(self):
        chanmap = self.chanmap()
        groups = [np.where(sk == chanmap)[0] for sk in np.unique(chanmap)]
        return groups

    def chanmap(self):
        stations = np.array([tr.stats.station for tr in self])
        unique = np.unique(stations)
        unique_dict = dict(zip(unique, np.arange(len(unique))))
        chanmap = np.array([unique_dict[chan] for chan in stations], dtype=int)
        return chanmap

    def write(self, filename, format='MSEED', **kwargs):

        from six import string_types
        f = filename

        if isinstance(filename, string_types):
            if filename.endswith('gz'):
                import gzip
                f = gzip.open(filename, 'w')
            elif filename.endswith('bz2'):
                import bz2
                f = bz2.BZ2File(filename, 'w')
            elif filename.endswith('zip'):
                print('Zip protocol is not supported')

        st_out = self.copy()

        return obsstream.Stream.write(st_out, f, format, **kwargs)

    write.__doc__ = obsstream.Stream.write.__doc__.replace('obspy',
                                                         'microquake')

    def valid(self, **kwargs):
        return is_valid(self, return_stream=True)

    def concat(self, comp_st):

        c = (comp_st is not None)
        if c:
            for i, (t1, t2) in enumerate(zip(comp_st.traces, self.traces)):
                self.detrend_norm(t2)
                comp_st.traces[i] = t1.__add__(t2, method=1, fill_value=0)
        else:
            for t in self:
                self.detrend_norm(t)

            comp_st = self

        return comp_st

    def unique_stations(self):

        return np.unique([tr.stats.station for tr in self])

    def plot(self, *args, **kwargs):
        """
        see Obspy stream.plot()
        """
        from microquake.imaging.waveform import WaveformPlotting
        waveform = WaveformPlotting(stream=self, *args, **kwargs)
        return waveform.plotWaveform(*args, **kwargs)


def is_valid(st_in, return_stream=False, STA=0.005, LTA=0.1, min_num_valid=5):
    """
        Determine if an event is valid or return valid traces in a  stream
        :param st_in: stream
        :type st_in: microquake.core.stream.Stream
        :param return_stream: return stream of valid traces if true else return
        true if the event is valid
        :type return_stream: bool
        :param STA: short term average used to determine if an event is valid
        :type STA: float
        :param LTA: long term average 
        :type LTA: float
        :param min_num_valid: minimum number of valid traces to declare the
        event valid
        :type min_num_valid: int
        :rtype: bool or microquake.core.stream.Stream
    """

    from scipy.ndimage.filters import gaussian_filter1d
    from microquake.signal.trigger import recursive_sta_lta

    st = st_in.copy()
    st.detrend('demean').detrend('linear')
    trstd = []
    trmax = []
    trs_out = []
    st_comp = composite_traces(st)
    for tr in st_comp:
        if not np.any(tr.data):
            continue
        sampling_rate = tr.stats.sampling_rate
        trstd.append(np.std(tr.data))
        trmax.append(np.max(np.abs(tr.data)))
        nsta = int(STA * sampling_rate)
        nlta = int(LTA * sampling_rate)
        # Tracer()()
        cft = recursive_sta_lta(np.array(tr.data), nsta, nlta)
        sfreq = tr.stats['sampling_rate']
        sigma = sfreq / (2 * np.pi * 100)
        cft = gaussian_filter1d(cft, sigma=sigma, mode='reflect')
        try:
            mx = np.r_[True, cft[1:] > cft[:-1]] & \
                 np.r_[cft[:-1] > cft[1:], True]
        except:
            continue

        i1 = np.nonzero(mx)[0]
        i2 = i1[cft[i1] > np.max(cft) / 2]
        try:
            tspan = (np.max(i2) - np.min(i2)) / sampling_rate
        except:
            Tracer()()

        ratio = np.max(np.abs(tr.data)) / np.std(tr.data)
        
        accept = True

        if len(i2) < 3:
            if ratio < 7.5:
                accept = False

        elif len(i2) > 4:
            accept = False
        else:
            if ratio < 12.5:
                accept = False
        if tspan > 0.1:
            accept = False  

        if (len(i2) == 2) and (tspan > 0.01) and (tspan < 0.1):
            if ratio > 5:
                accept = True

        if accept:
            trs_out.append(tr)

    st.traces = trs_out

    if return_stream:
        return st
    else:
        if len(st.unique_stations()) >= min_num_valid:
            return True
        else:
            return False


def check_for_dead_trace(tr):
    eps = 1e-6
    data = tr.data.copy()
    mean = np.mean(data)
    max = np.max(data) - mean
    min = np.min(data) - mean
    #print('%s: mean:%f max:%f min:%f' % (tr.get_id(), mean, max, min))
    if max < eps and np.abs(min) < eps:
        return 1
    else:
        return 0


def composite_traces(st):
    """
    Requires length and sampling_rates equal for all traces
    returns a new stream object containing composite trace for all station.
    The amplitude of the composite traces are the norm of the amplitude of
    the trace of all component and the phase of the trace (sign) is the sign
    of the first components of a given station.
    :param st: a stream object
    :type st: ~microquake.core.stream.Stream
    :rtype: ~microquake.core.stream.Stream

    """

    trsout = []

    for station in st.unique_stations():
        trs = st.select(station=station)

        if len(trs) == 1:
            trsout.append(trs[0].copy())
            continue
        npts = len(trs[0].data)
        buf = np.zeros(npts, dtype=trs[0].data.dtype)
        for tr in trs:
            dat = tr.data
            buf += (dat - np.mean(dat)) ** 2
        buf = np.sign(trs[0].data) * np.sqrt(buf)

        stats = trs[0].stats.copy()
        stats.channel = 'C'
        trsout.append(Trace(data=buf.copy(), header=stats))

    return Stream(traces=trsout)





# @uncompress
# def read(fname, format='MSEED', **kwargs):
#     if format in ENTRY_POINTS['mq-waveform'].keys():
#         format_ep = ENTRY_POINTS['mq-waveform'][format]
#         read_format = load_entry_point(format_ep.dist.key,
#                                    'microquake.plugin.waveform.%s' %
#                                        format_ep.name, 'readFormat')
#         return read_format(fname, **kwargs)
#     else:
#         return Stream(stream=obsstream.read(fname, format=format, **kwargs))


def read(fname, format='MSEED', **kwargs):
    if format in ENTRY_POINTS['waveform'].keys():
        format_ep = ENTRY_POINTS['waveform'][format]
        read_format = load_entry_point(format_ep.dist.key,'obspy.plugin.waveform.%s' %
                                       format_ep.name, 'readFormat')
        return Stream(stream=read_format(fname, **kwargs))
    else:
        return Stream(stream=obsstream.read(fname, format=format, **kwargs))

read.__doc__ = obsstream.read.__doc__.replace('obspy', 'microquake')
