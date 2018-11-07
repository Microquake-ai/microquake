"""
hack test for read-writing catalog
"""
from importlib import reload
import numpy as np
from datetime import datetime

from microquake.core.event import Arrival, Event, Origin, Magnitude, Pick
from microquake.core.event import ResourceIdentifier
from microquake.core.stream import Stream
from microquake.core import read
from microquake.core import UTCDateTime
from microquake.core.util import tools
from microquake.core import read_events
# from obspy import read
from obspy.core.event import QuantityError

import io
import base64


def write_read(event):
	xmlfile = 'event.xml'
	evformat = 'quakeml'
	event.write(xmlfile, format=evformat)
	event2 = read_events(xmlfile, format=evformat)[0]
	return event2


raydat = np.arange(12, dtype=float).reshape(3, 4)

sloc = np.array([651275., 4767395, -175])
time = UTCDateTime(datetime(2017, 1, 1))
# depth_errors = QuantityError(uncertainty=5)

pick = Pick(method='snr_picker', snr=3.3, time=time, phase_hint='P', trace_id='.003..C')
arv = Arrival(ray=raydat, phase='P', pick_id=pick.resource_id)

og = Origin(time=time, x=sloc[0], y=sloc[1], z=sloc[2], arrivals=[arv])

assert(og == og.copy())

assert(pick == pick.copy())
assert(arv == arv.copy())

og.arrivals = [arv]
ev = Event(origins=[og], picks=[pick])

# event.preferred_origin_id = origin.resource_id

ev2 = write_read(ev)
og2 = ev2.origins[0]


def check_key_match(obj1, obj2):
	print("Checking %s==%s" % (type(obj1), type(obj1)))
	keys1 = obj1.defaults.keys()
	keys2 = obj2.defaults.keys()
	assert(keys1 == keys2)
	vals = []
	for k in keys1:
		equal = obj1[k] == obj2[k]
		vals.append(equal)
		print(equal, "   key: %s" % (k))
		if not equal:
			print(obj1[k], " != ", obj2[k])
	if False in vals:
		print("FAILED")
	else:
		print("PASSED")


arv2 = og2.arrivals[0]

check_key_match(ev, ev2)

check_key_match(og, og2)
# check_key_match(og.arrivals[0], og2.arrivals[0])

# check_key_match(ev.origins[0], ev2.origins[0])

# check_key_match(ev.origins[0], ev2.origins[0])

check_key_match(arv.get_pick(), arv2.get_pick())

p2 = arv2.get_pick()
