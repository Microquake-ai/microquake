# import numpy as np
# import os
# import glob
from io import BytesIO
from microquake.core import Stream
from microquake.core import read
from microquake.core import read_events
from microquake.core.event import Event
# import datetime
import msgpack


EXTYPES = {'mseed': 0,
		  'quakeml': 1}


def pack(data):
	return msgpack.packb(data, default=_encode_one, use_bin_type=True)


def unpack(pack):
	return msgpack.unpackb(pack, ext_hook=_decode_one, raw=False)


def _encode_one(obj):
	if isinstance(obj, Stream):
		buf = BytesIO()
		obj.write(buf, format='mseed')
		return msgpack.ExtType(EXTYPES['mseed'], buf.getvalue())
	if isinstance(obj, Event):
		buf = BytesIO()
		obj.write(buf, format='quakeml')
		return msgpack.ExtType(EXTYPES['quakeml'], buf.getvalue())
	raise TypeError("Unknown type: %r" % (obj,))


def _decode_one(code, data):
	if code == EXTYPES['mseed']:
		return read(BytesIO(data))
	if code == EXTYPES['quakeml']:
		return read_events(BytesIO(data))
	return msgpack.ExtType(code, data)
