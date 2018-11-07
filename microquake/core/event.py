# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: event.py
#  Purpose: Expansion of the obspy.core.event module
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.event module

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import obspy.core.event as obsevent
from obspy.core import AttribDict
from obspy.core.event import ResourceIdentifier
import warnings
import numpy as np
import io
import base64
from copy import deepcopy
from obspy.core.event import *

import logging
logger = logging.getLogger()
log_level = logging.getLogger().getEffectiveLevel()

debug = False

class Event(obsevent.Event):

    extra_keys = ['ACCEPTED', 'ASSOC_SEISMOGRAM_NAMES', 'AUTO_PROCESSED',
                  'BLAST', 'CORNER_FREQUENCY', 'DYNAMIC_STRESS_DROP',
                  'ENERGY', 'ENERGY_P', 'ENERGY_S', 'EVENT_MODIFICATION_TIME',
                  'EVENT_NAME', 'EVENT_TIME_FORMATTED', 'EVENT_TIME_NANOS',
                  'LOCAL_MAGNITUDE', 'LOCATION_RESIDUAL', 'LOCATION_X',
                  'LOCATION_Y', 'LOCATION_Z', 'MANUALLY_PROCESSED',
                  'NUM_ACCEPTED_TRIGGERS', 'NUM_TRIGGERS', 'POTENCY',
                  'POTENCY_P', 'POTENCY_S', 'STATIC_STRESS_DROP', 'TAP_TEST',
                  'TEST', 'TRIGGERED_SITES', 'USER_NAME']

    __doc__ = obsevent.Event.__doc__.replace('obspy', 'microquake')

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)
        self.defaults['_format'] = None  # obsEvent read from quakeML contains this

    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def __str__(self):
        return "Event:\t%s\n\n%s" % (
            self.short_str(),
            "\n".join(super(Event, self).__str__().split("\n")[1:]))

    def short_str(self):
        out = ''
        if self.origins:
            og = self.preferred_origin() or self.origins[0]
            out += '%s | %s, %s, %s | %s' % (og.time, og.x, og.y, og.z, og.evaluation_mode)

        if self.magnitudes:
            magnitude = self.preferred_magnitude() or self.magnitudes[0]
            out += ' | %s %-2s' % (magnitude.mag,
                                   magnitude.magnitude_type)
        return out

class Origin(obsevent.Origin):
    __doc__ = obsevent.Origin.__doc__.replace('obspy', 'microquake')
    extra_keys = ['x', 'y', 'z', 'x_error', 'y_error', 'z_error']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)
            
    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)
    
    @property
    def loc(self):
        return np.array([self.x, self.y, self.z])

    @property
    def uncertainty(self):
        if self.origin_uncertainty is None:
            return None
        else:
            return self.confidence_ellipsoid.semi_major_axis_length

    def get_origin(self):
        if self.preferred_origin_id is not None:
            return self.preferred_origin_id.get_referred_object()

    def __str__(self, **kwargs):
        string = """
       resource_id: %s
              time: %s
                 x: %s
                 y: %s
                 z: %s
       uncertainty: %s
   evaluation_mode: %s
 evaluation_status: %s
                ---------
          arrivals: %d Elements
        """ \
            % (self.resource_id, self.time.strftime("%Y/%m/%d %H:%M:%S.%f"),
            self.x, self.y, self.z, self.uncertainty, self.evaluation_mode,
            self.evaluation_status,
            len(self.arrivals))
        return string


class Magnitude(obsevent.Magnitude):
    __doc__ = obsevent.Magnitude.__doc__.replace('obspy', 'microquake')
    extra_keys = ['corner_frequency', 'error']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)
            
    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)
       

class Pick(obsevent.Pick):
    __doc__ = obsevent.Pick.__doc__.replace('obspy', 'microquake')
    extra_keys = ['method', 'snr']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)
            
    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def __str__(self, **kwargs):

        string = """
       resource_id: %s
              time: %s
            method: %s
   evaluation_mode: %s
 evaluation_status: %s
        """ \
            % (self.resource_id, self.time.strftime("%Y/%m/%d %H:%M:%S.%f"),
            self.method, self.evaluation_mode,
            self.evaluation_status)
        return string


class Arrival(obsevent.Arrival):
    __doc__ = obsevent.Arrival.__doc__.replace('obspy', 'microquake')

    extra_keys = ['ray']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)
            
    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def get_pick(self):
        if self.pick_id is not None:
            return self.pick_id.get_referred_object()




def read_events(*args, **kwargs):

    cat = obsevent.read_events(*args, **kwargs)
    cat.events = [Event(ev) for ev in cat.events]
    return cat


def _init_handler(self, obspy_obj, **kwargs):
    """
    Handler to initialize microquake objects which
    inherit from ObsPy objects. If obspy_obj is none,
    Kwargs is expected to be a mix of obspy kwargs
    and microquake kwargs specified by the hardcoded
    extra_keys.
    """

    if obspy_obj and len(kwargs) > 0:
        raise AttributeError("Initialize from either \
                              obspy_obj or kwargs, not both")

    # default initialize the extra_keys args to None
    self['extra'] = {}
    [self.__setattr__(key, None) for key in self.extra_keys]

    if obspy_obj:
        _init_from_obspy_object(self, obspy_obj)
    else:
        extra_kwargs = pop_keys_matching(kwargs, self.extra_keys)
        super(type(self), self).__init__(**kwargs)  # init obspy_origin args
        [self.__setattr__(k, v) for k, v in extra_kwargs.items()]  # init extra_args


def _init_from_obspy_object(mquake_obj, obspy_obj):
    """
    When initializing microquake object from obspy_obj
    checks attributes for lists of obspy objects and
    converts them to equivalent microquake objects.
    """

    class_equiv = {obsevent.Pick: Pick,
                   obsevent.Arrival: Arrival,
                   obsevent.Origin: Origin,
                   obsevent.Magnitude: Magnitude}

    for key, val in obspy_obj.__dict__.items():
        if type(val) == list:
            out = []
            for item in val:
                itype = type(item)
                if itype in class_equiv:
                    out.append(class_equiv[itype](item))
                else:
                    out.append(item)
            mquake_obj.__setattr__(key, out)
        else:
            mquake_obj.__setattr__(key, val)


def _set_attr_handler(self, name, value, namespace='MICROQUAKE'):
    """
    Generic handler to set attributes for microquake objects
    which inherit from ObsPy objects. Assigns attributes not
    in default obspy object to extra args so they are correctly
    written to quakeML. Automatically loads any attributes in
    self['extra'] to regular attributes so microquake objects
    saved and reloaded from quakeml have extra attributes
    reassigned.
    """

    # parses extra args when constructing uquake from obspy
    if name in self.extra_keys:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self[name] = value
            if type(value) is np.ndarray:
                value = "npy64_" + array_to_b64(value)
            self['extra'][name] = {'value': value, 'namespace': namespace}

    elif name in self.defaults.keys():
        super(type(self), self).__setattr__(name, value)
    elif name == 'extra':
        if 'extra' not in self:  # hack for deepcopy to work
            self['extra'] = {}
        for key, adict in value.items():
            self.__setattr__(key, parse_string_val(adict.value))
    else:
        raise KeyError(name)
        

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def pop_keys_matching(dict_in, keys):
    # Move keys from dict_in to dict_out
    dict_out = {}
    for key in keys:
        if key in dict_in:
            dict_out[key] = dict_in.pop(key)
    return dict_out


def array_to_b64(array):
    output = io.BytesIO()
    np.save(output, array)
    content = output.getvalue()
    encoded = base64.b64encode(content).decode('utf-8')
    return encoded


def b64_to_array(b64str):
    arr = np.load(io.BytesIO(base64.b64decode(b64str)))
    return arr


def parse_string_val(val, arr_flag='npy64_'):
    """
    Parse extra args in quakeML which are all stored as string.
    """
    if val is None:  # hack for deepcopy ignoring isfloat try-except
        val = None
    elif isfloat(val):
        val = float(val)
    elif str(val) == 'None':
        val = None
    elif val[:len(arr_flag)] == 'npy64_':
        val = b64_to_array(val[len(arr_flag):])
    return val


class Ray:

    def __init__(self, nodes=[]):
        self.nodes = np.array(nodes)

    def length(self):
        if len(self.nodes) < 2:
            return 0

        length = 0
        for k, node1 in enumerate(self.nodes[0:-1]):
            node2 = self.nodes[k + 1]
            length += np.linalg.norm(node1 - node2)

        return length

    def __len__(self):
        return self.length()


def update_arrivals(origin, site):
    """
    This function calculates the distance, take off angle and azimuth for a
    set of arrivals
    :param site: a ~microquake.core.site object
    """
    import numpy as np

    oloc = origin.loc
    arrivals = origin.arrivals
    for k, arrival in enumerate(arrivals):
        pick = arrival.pick_id.get_referred_object()
        site_code = pick.waveform_id.station_code
        sloc = site.select(station=site_code).stations()[0].loc
        v_evt_sta = sloc - oloc
        arrivals[k].distance = np.linalg.norm(v_evt_sta)
        arrivals[k].azimuth = np.arctan2(v_evt_sta[0], v_evt_sta[1])\
                                   * 180 / np.pi
        hor = np.linalg.norm(v_evt_sta[0:2])
        arrivals[k].takeoff_angle = np.arctan2(hor, -v_evt_sta[2]) *\
                                         180 / np.pi


def break_down(event):
    origin = event.origins[0]
    print("break_down: Here's what obspy reads:")
    print(origin)
    print("origin res id: %s" % origin.resource_id.id)
    print("id(origin): %s" % id(origin))
    print("id(origin.resource_id):%s" % id(origin.resource_id))
    ref_obj = origin.resource_id.get_referred_object()
    print("id(ref_obj):%s" % id(ref_obj))

    return



# def make_pick(time, phase='P', wave_data=None, SNR=None, mode='automatic', status='preliminary', \
#               method_string=None, resource_id=None):

#     this_pick = Pick()
#     this_pick.time = time
#     this_pick.phase_hint = phase
#     this_pick.evaluation_mode = mode
#     this_pick.evaluation_status = status
#     if wave_data is not None:
#         this_pick.waveform_id = WaveformStreamID(
#             network_code=wave_data.stats.network,
#             station_code=wave_data.stats.station,
#             location_code=wave_data.stats.location,
#             channel_code=wave_data.stats.channel)
#     if SNR is not None:
#         #this_pick.comments = [Comment(text="SNR=%.3f" % SNR)]
#         if resource_id is not None:
#             this_pick.comments = [Comment(text="SNR=%.3f" % SNR, resource_id=resource_id)]
#         else:
#             this_pick.comments = [Comment(text="SNR=%.3f" % SNR, force_resource_id=False)]

#     if method_string is not None:
#         method = AttribDict()
#         method['namespace'] = 'MICROQUAKE'
#         method['value'] = method_string
#         this_pick['extra'] = AttribDict()
#         this_pick['extra']['method'] = method

#     return this_pick
