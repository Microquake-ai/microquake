from io import BytesIO

from microquake.core import read, read_events
from microquake.db.connectors import connect_redis

redis_connector = connect_redis()


def set_event(event_id, catalogue=None, fixed_length=None, context=None,
              variable_length=None, ttl=10800):
    event = redis_connector.Hash(event_id)

    if catalogue is not None:
        file_out = BytesIO()
        catalogue.write(file_out, format='quakeml')
        event.update(catalogue=file_out.getvalue())

    if fixed_length is not None:
        file_out = BytesIO()
        fixed_length.write(file_out, format='mseed')
        event.update(fixed_length=file_out.getvalue())

    if context is not None:
        file_out = BytesIO()
        context.write(file_out, format='mseed')
        event.update(context=file_out.getvalue())

    if variable_length is not None:
        file_out = BytesIO()
        variable_length.write(file_out, format='mseed')
        event.update(variable_length=file_out.getvalue())

    event.expire(ttl)

    return event


def get_event(event_id):
    event = redis_connector.Hash(event_id)

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
