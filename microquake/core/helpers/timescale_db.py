from microquake.core.stream import Trace, Stream
from microquake.core.settings import settings
import numpy as np
from obspy.core import UTCDateTime
from loguru import logger
from microquake.db.models.alchemy import ContinuousData
from microquake.db.connectors import connect_timescale
from datetime import datetime


def get_continuous_data(start_time, end_time, sensor_id=None):

    if type(start_time) is datetime:
        start_time = UTCDateTime(start_time)

    if type(end_time) is datetime:
        end_time = UTCDateTime(end_time)

    session = connect_timescale()
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
        if not np.all(cd.x == 0):
            tr_x = Trace(data=np.array(cd.x))
            tr_x.stats.starttime = UTCDateTime(cd.time)
            tr_x.stats.sampling_rate = cd.sample_rate
            tr_x.stats.channel = 'X'
            tr_x.stats.station = str(cd.sensor_id)
            tr_x.stats.network = network_code
            traces.append(tr_x)
        if not np.all(cd.y == 0):
            tr_y = Trace(data=np.array(cd.y))
            tr_y.stats.starttime = UTCDateTime(cd.time)
            tr_y.stats.sampling_rate = cd.sample_rate
            tr_y.stats.channel = 'Y'
            tr_y.stats.station = str(cd.sensor_id)
            tr_y.stats.network = network_code
            traces.append(tr_y)
        if not np.all(cd.z == 0):
            tr_z = Trace(data=np.array(cd.z))
            tr_z.stats.starttime = UTCDateTime(cd.time)
            tr_z.stats.sampling_rate = cd.sample_rate
            tr_z.stats.channel = 'Z'
            tr_z.stats.station = str(cd.sensor_id)
            tr_z.stats.network = network_code
            traces.append(tr_z)

    st = Stream(traces=traces).trim(starttime=start_time,
                                    endtime=end_time)

    session.close()
    return st
