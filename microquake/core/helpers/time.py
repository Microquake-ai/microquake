import pytz
from dateutil.tz import tzoffset
from microquake.core.settings import settings


def get_time_zone():
    """
    returns a time zone compatible object Handling of time zone is essential
    for seismic system as UTC time is used in the as the default time zone
    :return: a time zone object
    """

    tz_settings = settings.get('time_zone')

    if tz_settings.type == "UTC_offset":
        offset = float(tz_settings.offset)    # offset in hours
        tz_code = tz_settings.time_zone_code  # code for the time zone
        tz = tzoffset(tz_code, offset * 3600)

    elif tz_settings.type == "time_zone":
        valid_time_zones = pytz.all_timezones

        if tz_settings.time_zone_code not in valid_time_zones:
            # raise an exception
            pass
        else:
            # TODO: incompatible assignment
            tz = pytz.timezone(tz_settings.time_zone_code)

    return tz
