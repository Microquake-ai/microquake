from functools import wraps

from libs.microquake.microquake.db.models.redis import get_event


def deserialize_message(func):
    @wraps(func)
    def wrapper(*args, event_id, **kwargs):
        dict_in = get_event(event_id)

        return func(**dict_in)

    return wrapper
