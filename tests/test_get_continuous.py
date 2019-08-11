# this test requires a connection to an IMS system

"""
get_continuous(base_url, start_datetime, end_datetime,
                   site_ids, format='binary-gz', network='',
                   sampling_rate=6000., nan_limit=10):
"""

from microquake.clients.ims import web_client
from microquake.core import UTCDateTime
from importlib import reload
reload(web_client)

base_url = 'http://10.95.74.35:8002/ims-database-server/databases/mgl'
station_ids = ['20']
start_time = UTCDateTime.now() - 10 * 60
end_time = UTCDateTime.now() - 8 * 60

st = web_client.get_continuous(base_url, start_time, end_time, station_ids)



