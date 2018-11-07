# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: serializer.py
#  Purpose: utility to encode and decode object using base64
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

def encode_pickle(obj):
    """Encodes any python object as a
    :py:mod:`base64` string.
    """
    import pickle
    import bz2

    buf = pickle.dumps(obj)
    comp_ser_obj = bz2.compress(buf)

    return comp_ser_obj


def decode_pickle(compressed_obj):
    """Decodes a :py:mod:`base64` string into a
    :py:class:`obspy.core.stream.Stream`.
    """
    import pickle
    import bz2

    ser_obj = bz2.decompress(compressed_obj)
    obj = pickle.loads(ser_obj)
    return obj


def encode_base64(buffered_object):
    """
    Encode an event for storage in the database
    :param event: a microquake.core.event object
    :return: encoded event object in compressed (bz2) format
    """
    from bz2 import compress
    from base64 import b64encode

    return b64encode(compress(buffered_object))


def decode_base64(encoded_object):
    """
    decode an event stored in the database
    :param encoded_event: compressed serialized object stored in the DB
    :return: microquake.core.event.event
    """

    from bz2 import decompress
    from base64 import b64decode

    return decompress(b64decode(encoded_object))
