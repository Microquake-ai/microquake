"""
Graveyard for deprecated json
"""

from obspy import UTCDateTime
from microquake.core import Trace, Stream
from obspy.core.util.attribdict import AttribDict
import numpy as np


def trace_from_data(stats, data_list):
    trc = Trace()
    trc.stats = AttribDict(stats)
    trc.data = np.array(data_list, dtype=np.float32)
    return trc


def trace_from_json(trace_json_object):
    trace_json_object['stats']['starttime'] = UTCDateTime(int(trace_json_object['stats']['starttime']) / 1e9)
    trace_json_object['stats']['endtime'] = UTCDateTime(int(trace_json_object['stats']['endtime']) / 1e9)
    trc = Trace.create_from_data(stats=trace_json_object['stats'], data_list=trace_json_object['data'])
    return trc


def trace_to_json(trace):
    trace_dict = dict()
    trace_dict['stats'] = dict()
    for key in trace.stats.keys():
        if isinstance(trace.stats[key], UTCDateTime):
            trace_dict['stats'][key] = int(np.float64(trace.stats[key].timestamp) * 1e9)
        else:
            trace_dict['stats'][key] = trace.stats[key]

    trace_dict['data'] = trace.data.tolist()
    return trace_dict


def stream_from_json_traces(traces_json_list):
    return Stream(traces=[trace_from_json(tr) for tr in traces_json_list])


def stream_to_json_traces(stream):
    return [trace_to_json(tr) for tr in stream]
