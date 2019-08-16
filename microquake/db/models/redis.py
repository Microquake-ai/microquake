from io import BytesIO
from time import time

import rom.util
from microquake.core import read, read_events
from microquake.db.connectors import connect_redis
from rom import Float, Model, String

rom.util.get_connection = connect_redis


class Event(Model):
    event_id = String(required=True, unique=True, suffix=True)
    catalogue = String()
    fixed_length = String()
    context = String()
    variable_length = String()
    created_at = Float(default=time)

    valid_until = Float(index=True)

    def expire(self, ttl, write_now=False):
        self.valid_until = time.time() + ttl

        if write_now:
            self.save()

    @classmethod
    def expire_old(cls):
        for old in cls.query.filter(valid_until=(None, time.time())).all():
            old.delete()


def set_event(event_id, catalogue=None, fixed_length=None, context=None,
              variable_length=None, timeout=10800):
    Event.expire_old()
    event = Event(event_id=event_id)
    event.expire(timeout)

    if catalogue is not None:
        file_out = BytesIO()
        catalogue.write(file_out, format='quakeml')
        event['catalogue'] = file_out.getvalue()

    if fixed_length is not None:
        file_out = BytesIO()
        fixed_length.write(file_out, format='mseed')
        event['fixed_length'] = file_out.getvalue()

    if context is not None:
        file_out = BytesIO()
        context.write(file_out, format='mseed')
        event['context'] = file_out.getvalue()

    if variable_length is not None:
        file_out = BytesIO()
        variable_length.write(file_out, format='mseed')
        event['variable_length'] = file_out.getvalue()

    event.save()

    return event


def get_event(event_id):
    event = Event.get_by(event_id=event_id)

    dict_in = {}

    if b'catalogue' in event.keys():
        bytes = event[b'catalogue']
        dict_in['catalogue'] = read_events(BytesIO(bytes), format='quakeml')

    if b'fixed_length' in event.keys():
        bytes = event[b'fixed_length']
        dict_in['fixed_length'] = read(BytesIO(bytes), format='mseed')

    if b'context' in event.keys():
        bytes = event[b'context']
        dict_in['context'] = read(BytesIO(bytes), format='mseed')

    if b'variable_length' in event.keys():
        bytes = event[b'variable_length']
        dict_in['variable_length'] = read(BytesIO(bytes),
                                          format='mseed')

    return dict_in
