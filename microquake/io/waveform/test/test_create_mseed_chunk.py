from microquake.io.waveform import core
from importlib import reload
from microquake.core import Trace, Stream
import numpy as np
from pandas import DataFrame
from spp.utils.kafka import KafkaHandler
from logging import getLogger

# kafka_brokers = CONFIG.DATA_CONNECTOR['kafka']['brokers']
kafka_brokers = ['localhost:9092']
kafka_topic = 'test'
# kafka_topic = CONFIG.DATA_CONNECTOR['kafka']['topic']

kafka_handler = KafkaHandler(kafka_brokers)

logger = getLogger(__name__)

reload(core)

# create data
trs = []
for i in np.arange(10):
    tr = Trace(data=np.random.randn(1000))
    tr.stats.network = 'TEST'
    tr.stats.station = '%d' % i
    tr.stats.channel = 'X'
    trs.append(tr)

st = Stream(traces=trs)

result = core.create_mseed_chunk(st)

df = DataFrame(result)

df_grouped = df.groupby(['key'])

logger.debug("Grouped DF Stats:" + str(df_grouped.size()))

chunks = []
for name, group in df_grouped:
    data = b''
    for g in group['blob'].values:
        data += g
    timestamp = int(name.timestamp() * 1e3)
    key = name.strftime('%Y-%d-%m %H:%M:%S.%f').encode('utf-8')
    kafka_handler.send_to_kafka(topic=data, key=key, message=data)
kafka_handler.producer.flush()


# etime = time.time() - stime
# logger.info("==> Inserted stream chunks into Kafka in: %.2f" % etime)


