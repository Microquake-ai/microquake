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

from __future__ import absolute_import, division, print_function, unicode_literals

import base64
import io
import logging
import warnings
from copy import deepcopy

import numpy as np
import obspy.core.event as obsevent
from obspy.core import AttribDict
from obspy.core.event import *
from obspy.core.event import ResourceIdentifier

logger = logging.getLogger()
log_level = logging.getLogger().getEffectiveLevel()

debug = False


class Event(obsevent.Event):

    # _format keyword is actualy a missing obspy default
    extra_keys = ['_format', 'ACCEPTED', 'ASSOC_SEISMOGRAM_NAMES', 'AUTO_PROCESSED',
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

        self.picks += picks


class Origin(obsevent.Origin):
    __doc__ = obsevent.Origin.__doc__.replace('obspy', 'microquake')
    extra_keys = ['x', 'y', 'z', 'x_error', 'y_error', 'z_error', 'scatter',
                  'interloc_vmax', 'interloc_time']

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
            return self.origin_uncertainty.confidence_ellipsoid\
                .semi_major_axis_length

    def get_origin(self):
        if self.preferred_origin_id is not None:
            return self.preferred_origin_id.get_referred_object()

    def get_all_magnitudes_for_origin(self, cat):
        magnitudes = []

        for event in cat:
            for mag in event.magnitudes:
                if mag.origin_id.id == self.resource_id.id:
                    magnitudes.append(mag)

        return magnitudes

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
    extra_keys = ['method', 'snr', 'trace_id']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)
        # MTH  - this seems to have been left out ??
        if obspy_obj:
            wid = self.waveform_id
            self.trace_id = "%s.%s.%s.%s" % (wid.network_code, wid.station_code, wid.location_code, wid.channel_code)

    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def __str__(self, **kwargs):
        string = """
          trace_id: %s
              time: %s
             phase:[%s]
            method: %s [%s]
   evaluation_mode: %s
 evaluation_status: %s
       resource_id: %s
        """ \
            % (self.trace_id, self.time.strftime("%Y/%m/%d %H:%M:%S.%f"),
               self.phase_hint, self.method, self.snr, self.evaluation_mode,
               self.evaluation_status, self.resource_id)
        return string

    def get_sta(self):
        if self.waveform_id is not None:
            return self.waveform_id.station_code


class Arrival(obsevent.Arrival):
    __doc__ = obsevent.Arrival.__doc__.replace('obspy', 'microquake')

    # extra_keys = ['ray', 'backazimuth', 'inc_angle']
    extra_keys = ['ray', 'backazimuth', 'inc_angle', 'polarity',
                  'peak_vel', 'tpeak_vel', 't1', 't2', 'pulse_snr',
                  'peak_dis', 'tpeak_dis', 'max_dis', 'tmax_dis',
                  'dis_pulse_width', 'dis_pulse_area',
                  'smom', 'fit', 'tstar',
                  'hypo_dist_in_m',
                  'vel_flux', 'vel_flux_Q', 'energy',
                  'fmin', 'fmax',
                  'traces',
                  ]

    # extra_keys = ['ray', 'backazimuth', 'inc_angle', 'velocity_pulse', 'displacement_pulse']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)

    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def get_pick(self):
        if self.pick_id is not None:
            return self.pick_id.get_referred_object()


def get_arrival_from_pick(arrivals, pick):
    """
      return arrival corresponding to pick

      :param arrivals: list of arrivals
      :type arrivals: list of either obspy.core.event.origin.Arrival
                      or microquake.core.event.origin.Arrival
      :param pick: P or S pick
      :type pick: either obspy.core.event.origin.Pick
                      or microquake.core.event.origin.Pick
      :return arrival
      :rtype: obspy.core.event.origin.Arrival or
              microquake.core.event.origin.Arrival
    """

    arrival = None
    for arr in arrivals:
        if arr.pick_id == pick.resource_id:
            arrival = arr
            break

    return arrival


def read_events(*args, **kwargs):

    cat = obsevent.read_events(*args, **kwargs)
    cat.events = [Event(ev) for ev in cat.events]
    return cat


def _init_handler(self, obspy_obj, **kwargs):
    """
    Handler to initialize microquake objects which
    inherit from ObsPy class. If obspy_obj is none,
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
    which inherit from ObsPy objects. If 'name' is not in
    default keys then it will be set in self['extra'] dict. If
    'name' is not in default keys but in the self.extra_keys
    then it will also be set as a class attribute. When loading
    extra keys from quakeml file, those in self.extra_keys will
    be set as attributes.
    """

    #  use obspy default setattr for default keys
    if name in self.defaults.keys():
        super(type(self), self).__setattr__(name, value)
    elif name in self.extra_keys:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self[name] = value
        if type(value) is np.ndarray:
            value = "npy64_" + array_to_b64(value)
        elif type(value) is str:
            if "npy64_" in value:
                value.replace("npy64_", "")
                b64_to_array(value)
        self['extra'][name] = {'value': value, 'namespace': namespace}
    # recursive parse of 'extra' args when constructing uquake from obspy
    elif name == 'extra':
        if 'extra' not in self:  # hack for deepcopy to work
            self['extra'] = {}
        for key, adict in value.items():
            if key in self.extra_keys:
                self.__setattr__(key, parse_string_val(adict.value))
            else:
                self['extra'][key] = adict
    else:
        raise KeyError(name)


def _set_attr_handler2(self, name, value, namespace='MICROQUAKE'):
    """
    Generic handler to set attributes for microquake objects
    which inherit from ObsPy objects. If 'name' is not in
    default keys then it will be set in self['extra'] dict. If
    'name' is not in default keys but in the self.extra_keys
    then it will also be set as a class attribute. When loading
    extra keys from quakeml file, those in self.extra_keys will
    be set as attributes.
    """

    #  use obspy default setattr for default keys
    if name in self.defaults.keys():
        super(type(self), self).__setattr__(name, value)
    # recursive parse of extra args when constructing uquake from obspy
    elif name == 'extra':
        if 'extra' not in self:  # hack for deepcopy to work
            self['extra'] = {}
        for key, adict in value.items():
            self.__setattr__(key, parse_string_val(adict.value))
    else:  # branch for extra keys
        if name in self.extra_keys:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self[name] = value
        if type(value) is np.ndarray:
            value = "npy64_" + array_to_b64(value)
        self['extra'][name] = {'value': value, 'namespace': namespace}


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


# MTH: this could(should?) be moved to waveforms/pick.py ??
def make_pick(time, phase='P', wave_data=None, snr=None, mode='automatic',
              status='preliminary', method_string=None, resource_id=None):

    this_pick = Pick()
    this_pick.time = time
    this_pick.phase_hint = phase
    this_pick.evaluation_mode = mode
    this_pick.evaluation_status = status

    this_pick.method = method_string
    this_pick.snr = snr

    if wave_data is not None:
        this_pick.waveform_id = WaveformStreamID(
            network_code=wave_data.stats.network,
            station_code=wave_data.stats.station,
            location_code=wave_data.stats.location,
            channel_code=wave_data.stats.channel)

        this_pick.trace_id = wave_data.get_id()

    return this_pick
