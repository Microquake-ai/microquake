from datetime import datetime
from time import time

import numpy as np
import sqlalchemy as db
from obspy.core import UTCDateTime
from pytz import utc
from sqlalchemy.orm import sessionmaker

from loguru import logger
from microquake.core.settings import settings
from microquake.core.stream import Stream, Trace
from microquake.db.models.alchemy import Recording, metadata, processing_logs
from redis import ConnectionPool, Redis
from rq import Queue
from walrus import Walrus


def connect_redis():
    return RedisWrapper().redis_connect(url=settings.REDIS_WALRUS_URL)


class RedisWrapper(object):
    shared_state = {}

    def __init__(self):
        self.__dict__ = self.shared_state

    def redis_connect(self, url):
        try:
            self.connection_pool
        except AttributeError:
            self.connection_pool = ConnectionPool.from_url(url)

        return Walrus(connection_pool=self.connection_pool)


class RedisQueue:
    def __init__(self, queue, timeout=1800):
        self.redis = Redis.from_url(url=settings.REDIS_RQ_URL)
        self.timeout = timeout
        self.queue = queue
        self.rq_queue = Queue(self.queue, connection=self.redis,
                              default_timeout=self.timeout)

    def submit_task(self, func, *args, **kwargs):
        return self.rq_queue.enqueue(func, *args, **kwargs)


# def submit_task_to_rq(queue, func, *args, **kwargs):
#     with connect_redis() as redis:
#         rq_queue = Queue(queue, connection=redis)
#         return rq_queue.enqueue(func, *args, **kwargs)

# rq worker --url redis://redisdb:6379 --log-format '%(asctime)s '  api

def connect_rq(message_queue):
    redis = connect_redis()

    return Queue(message_queue, connection=redis)


def connect_postgres():

    db_name = settings.POSTGRES_DB_NAME
    postgres_url = settings.POSTGRES_URL + db_name

    engine = db.create_engine(postgres_url)
    connection = engine.connect()
    # Create tables if they do not exist
    metadata.create_all(engine)

    return connection


def create_postgres_session():

    db_name = settings.POSTGRES_DB_NAME
    postgres_url = settings.POSTGRES_URL + db_name

    engine = db.create_engine(postgres_url)
    pg = connect_postgres()
    Session = sessionmaker(bind=engine)

    return Session()


def get_continuous_data(starttime, endtime, sensor_id=None):

    session = create_postgres_session()

    t0 = time()

    if sensor_id is not None:
        session.query(Recording).filter(
            Recording.time <= endtime).filter(
            Recording.end_time >= starttime).filter(
            Recording.sensor_id == sensor_id).all()

    else:
        results = session.query(Recording).filter(
            Recording.time <= endtime).filter(
            Recording.end_time >= starttime).all()

    t1 = time()
    logger.info('retrieving the data in {} seconds'.format(t1 - t0))

    trs = []

    for trace in results:
        tr = Trace()

        for channel in ['x', 'y', 'z']:

            if np.all(trace.__dict__[channel] == 0):
                continue

            tr.stats.network = settings.NETWORK_CODE
            tr.stats.station = str(trace.sensor_id)
            tr.stats.location = ''
            tr.stats.channel = channel
            tr.stats.sampling_rate = trace.sample_rate
            tr.stats.starttime = UTCDateTime(trace.time)
            tr.data = np.array(trace.__dict__[channel])
            trs.append(tr)

    st = Stream(traces=trs)
    st.trim(starttime=UTCDateTime(starttime), endtime=UTCDateTime(endtime),
            pad=False, fill_value=0)

    return st


def record_processing_logs_pg(event, status, processing_step,
                              processing_step_id, processing_time_second):
    """
    Record the processing logs in the postgres database
    :param event: event being processed
    :param status: processing status (accepted values are success, failed)
    :param processing_step: processing step name
    :param processing_step_id: processing step identifier integer
    :param processing_time_second: processing dealy for this step in seconds
    :return:
    """

    if 'catalog' in str(type(event)).lower():
        event = event[0]

    origin = event.preferred_origin()

    if origin is None:
        origin = event.origins[-1]

    event_time = origin.time.datetime.replace(tzinfo=utc)

    current_time = datetime.utcnow().replace(tzinfo=utc)
    processing_delay_second = (current_time - event_time).total_seconds()

    document = {'event_id': event.resource_id.id,
                'event_timestamp': event_time,
                'processing_timestamp': current_time,
                'processing_step_name': processing_step,
                'processing_step_id': processing_step_id,
                'processing_delay_second': processing_delay_second,
                'processing_time_second': processing_time_second,
                'processing_status': status}

    with connect_postgres() as pg:
        query = db.insert(processing_logs)
        values_list = [document]

        result = pg.execute(query, values_list)

        pg.close()

    return result
