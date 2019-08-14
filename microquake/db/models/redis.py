from io import BytesIO

from microquake.core import read, read_events
from pottery import RedisDict
from microquake.db.connectors import connect_redis


def set_event(event_id, catalogue=None, fixed_length=None, context=None,
              variable_length=None, timeout=10800):
    redis = connect_redis()
    dict_out = RedisDict(redis=redis, key=event_id)

    if catalogue is not None:
        file_out = BytesIO()
        catalogue.write(file_out, format='quakeml')
        dict_out['catalogue'] = file_out.getvalue()

    if fixed_length is not None:
        file_out = BytesIO()
        fixed_length.write(file_out, format='mseed')
        dict_out['fixed_length'] = file_out.getvalue()

    if context is not None:
        file_out = BytesIO()
        context.write(file_out, format='mseed')
        dict_out['context'] = file_out.getvalue()

    if variable_length is not None:
        file_out = BytesIO()
        variable_length.write(file_out, format='mseed')
        dict_out['variable_length'] = file_out.getvalue()

    redis.expire(event_id, timeout)
    return dict_out


def get_event(event_id):
    redis = connect_redis()
    serialized_dict_in = RedisDict(redis=redis, key=event_id)

    dict_in = {}

    if b'catalogue' in serialized_dict_in.keys():
        bytes = serialized_dict_in[b'catalogue']
        dict_in['catalogue'] = read_events(BytesIO(bytes), format='quakeml')

    if b'fixed_length' in serialized_dict_in.keys():
        bytes = serialized_dict_in[b'fixed_length']
        dict_in['fixed_length'] = read(BytesIO(bytes), format='mseed')

    if b'context' in serialized_dict_in.keys():
        bytes = serialized_dict_in[b'context']
        dict_in['context'] = read(BytesIO(bytes), format='mseed')

    if b'variable_length' in serialized_dict_in.keys():
        bytes = serialized_dict_in[b'variable_length']
        dict_in['variable_length'] = read(BytesIO(bytes),
                                          format='mseed')

    return dict_in
