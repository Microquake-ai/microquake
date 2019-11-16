from microquake.core.stream import Trace, Stream
from microquake.core.settings import settings
import numpy as np
from obspy.core import UTCDateTime
from loguru import logger
from microquake.db.models.alchemy import ContinuousData
from microquake.db.connectors import connect_timescale
from datetime import datetime
from sqlalchemy import desc
from pytz import utc


def get_continuous_data(start_time, end_time, sensor_id=None):

    db_lag = get_db_lag()
    logger.info(f'the database lag is {db_lag} seconds')

    if type(start_time) is datetime:
        start_time = UTCDateTime(start_time)

    if type(end_time) is datetime:
        end_time = UTCDateTime(end_time)

    session, engine = connect_timescale()
    inventory = settings.inventory

    e_time = end_time.datetime
    s_time = start_time.datetime

    network_code = inventory.networks[0].code

    t = ContinuousData.time
    et = ContinuousData.end_time
    sid = ContinuousData.sensor_id

    if sensor_id is None:
        logger.info('requesting data for all sensors')
        cds = session.query(ContinuousData).filter(t <= e_time,
                                                   et > s_time)
    else:
        if inventory.select(sensor_id) is None:
            logger.error(f'the sensor {sensor_id} is not in the inventory')

            return
        logger.info(f'requesting data for sensor {sensor_id}')
        cds = session.query(ContinuousData).filter(t <= e_time,
                                                   et > s_time,
                                                   sid == sensor_id)

    traces = []
    for cd in cds:
        x = np.array(cd.x)
        y = np.array(cd.y)
        z = np.array(cd.z)
        tr_x = Trace(data=x)
        tr_x.stats.starttime = UTCDateTime(cd.time)
        tr_x.stats.sampling_rate = cd.sample_rate
        tr_x.stats.channel = 'X'
        tr_x.stats.station = str(cd.sensor_id)
        tr_x.stats.network = network_code
        traces.append(tr_x)
        tr_y = Trace(data=y)
        tr_y.stats.starttime = UTCDateTime(cd.time)
        tr_y.stats.sampling_rate = cd.sample_rate
        tr_y.stats.channel = 'Y'
        tr_y.stats.station = str(cd.sensor_id)
        tr_y.stats.network = network_code
        traces.append(tr_y)
        tr_z = Trace(data=z)
        tr_z.stats.starttime = UTCDateTime(cd.time)
        tr_z.stats.sampling_rate = cd.sample_rate
        tr_z.stats.channel = 'Z'
        tr_z.stats.station = str(cd.sensor_id)
        tr_z.stats.network = network_code
        traces.append(tr_z)

    time_now = UTCDateTime.now()
    delay = time_now - end_time

    st = Stream(traces=traces).trim(starttime=start_time, endtime=end_time)

    if st is None:
        logger.warning(f'no data recovered from the database! '
                       f'the current database lag is {db_lag}')
        return None

    duration = (end_time - start_time)

    trs = []
    # st = st.merge(fill_value=np.nan)
    for i, tr in enumerate(st):
        expected_number_sample = tr.stats.sampling_rate * duration
        if np.all(tr.data == 0):
            logger.warning(f'data from sensor {tr.stats.station} contains '
                           f'only zero. The trace will not be kept')
            continue
        elif np.any(np.isnan(tr.data)):
            logger.warning(f'data from sensor {tr.stats.station} contains '
                           f'some NaN. The trace will not be kept')
            continue
        elif len(tr) < 0.9 * expected_number_sample:
            logger.warning(f'data from sensor {tr.stats.station} contains '
                           f'too little data. The trace will not be kepy')
            continue

        trs.append(tr)

    if not trs:
        return None

    st = Stream(traces=trs).trim(starttime=start_time,
                                 endtime=end_time)

    session.close()
    engine.dispose()

    return st.detrend('demean')


def get_db_lag(percentile=75):
    """
    returns lag in seconds
    :param percentile: percentile of data to use to determine the delay.
    For instance, if percentile = 75, 75 percent of the trace will have a lag
    of less than the returned value.
    :return: lag in second
    """

    session, engine = connect_timescale()

    inventory = settings.inventory
    t = ContinuousData.time
    sensor_id = ContinuousData.sensor_id

    times = []
    for sensor in inventory.stations():

        records = session.query(t, sensor_id).filter(
            sensor_id == sensor.code).order_by(desc(t)).limit(1)

        for record in records:
            times.append(record.time.timestamp())

    if not times:
        return None

    time = datetime.utcfromtimestamp(np.percentile(times, percentile))

    lag = datetime.utcnow().replace(tzinfo=utc) - time.replace(tzinfo=utc)

    session.close()
    engine.dispose()

    return lag.total_seconds()


