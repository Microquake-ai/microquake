# import pytest
# from tests.helpers.data_utils import get_test_data

from io import BytesIO
from microquake.core.settings import settings
import requests
from microquake.clients import api_client
from microquake.core import read_events

import msgpack

# from spp.pipeline.interactive_pipeline import interactive_pipeline
base_url = settings.get('api_base_url')

test_data_name = "test_interactive_pipeline"

event_id = "smi:local/2019/06/27/08/25_47_115750099.e"
re = api_client.get_event_by_id(base_url, event_id)

event_bytes = requests.get(re.event_file).content
mseed_bytes = requests.get(re.waveform_file).content

picks = '{"event_resource_id":"smi:local/2019/06/27/08/25_47_115750099.e' \
        '","data":[{"arrival_resource_id":"smi:local/ffbe62af-4b48-49ae' \
        '-9be3-36e236a30ad5",' \
        '"event":"smi:local/2019/06/27/08/25_47_115750099.e",' \
        '"origin":"smi:local/0a49d7b8-1919-4ba6-aaaa-e835535ba0dd",' \
        '"pick":{"pick_resource_id":"smi:local/84f3535f-ce76-4e0e-a048' \
        '-49f637f18e58",' \
        '"event":"smi:local/2019/06/27/08/25_47_115750099.e","site":1,' \
        '"network":1,"station":37,' \
        '"time_utc":"2019-06-27T08:25:48.021550Z","time_errors":null,' \
        '"method_id":null,"filter_id":null,"onset":null,' \
        '"phase_hint":"P","polarity":null,"evaluation_mode":"automatic",' \
        '"evaluation_status":"preliminary"},"site":1,"network":1,' \
        '"phase":"P","time_correction":null,"azimuth":37.0213943204641,' \
        '"distance":1566.39623925138,"takeoff_angle":100.520098920753,' \
        '"time_residual":-0.012353,' \
        '"earth_model":"smi:local/f9faa611-fcd7-42d4-908d-e5b327e42bf6' \
        '"},{"pick":{"site":1,"network":1,"station":78,' \
        '"time_utc":"2019-06-27T08:25:47.889916Z","time_errors":null,' \
        '"method_id":null,"filter_id":null,"onset":null,' \
        '"phase_hint":"P","polarity":null,"evaluation_mode":"manual",' \
        '"evaluation_status":null},"site":1,"network":1,"phase":"P",' \
        '"time_correction":null,"azimuth":null,"distance":null,' \
        '"takeoff_angle":null,"time_residual":null,"earth_model":null},' \
        '{"arrival_resource_id":"smi:local/ff1ea391-a6ff-48ff-aee9' \
        '-95ea70d3359a",' \
        '"event":"smi:local/2019/06/27/08/25_47_115750099.e",' \
        '"origin":"smi:local/0a49d7b8-1919-4ba6-aaaa-e835535ba0dd",' \
        '"pick":{"pick_resource_id":"smi:local/3d185b2a-360f-44fa-9ed6' \
        '-9e2be6e6e207",' \
        '"event":"smi:local/2019/06/27/08/25_47_115750099.e","site":1,' \
        '"network":1,"station":9,' \
        '"time_utc":"2019-06-27T08:25:47.994092Z","time_errors":null,' \
        '"method_id":null,"filter_id":null,"onset":null,' \
        '"phase_hint":"P","polarity":null,"evaluation_mode":"automatic",' \
        '"evaluation_status":"preliminary"},"site":1,"network":1,' \
        '"phase":"P","time_correction":null,"azimuth":244.753316549404,' \
        '"distance":1337.30104769653,"takeoff_angle":92.0077229506251,' \
        '"time_residual":0.007648,' \
        '"earth_model":"smi:local/f9faa611-fcd7-42d4-908d-e5b327e42bf6' \
        '"},{"azimuth":null,"distance":null,"earth_model":null,' \
        '"phase":"P","pick":{"evaluation_mode":"manual",' \
        '"phase_hint":"P","station":"114",' \
        '"time_utc":"2019-06-27T08:25:47.389011Z",' \
        '"evaluation_status":null,"time_errors":null,"method_id":null,' \
        '"filter_id":null,"onset":null,"polarity":null},' \
        '"time_correction":null,"time_residual":null,' \
        '"takeoff_angle":null},{"azimuth":null,"distance":null,' \
        '"earth_model":null,"phase":"P","pick":{' \
        '"evaluation_mode":"manual","phase_hint":"P","station":"119",' \
        '"time_utc":"2019-06-27T08:25:47.381685Z",' \
        '"evaluation_status":null,"time_errors":null,"method_id":null,' \
        '"filter_id":null,"onset":null,"polarity":null},' \
        '"time_correction":null,"time_residual":null,' \
        '"takeoff_angle":null},{"azimuth":null,"distance":null,' \
        '"earth_model":null,"phase":"S","pick":{' \
        '"evaluation_mode":"manual","phase_hint":"S","station":"41",' \
        '"time_utc":"2019-06-27T08:25:47.500733Z",' \
        '"evaluation_status":null,"time_errors":null,"method_id":null,' \
        '"filter_id":null,"onset":null,"polarity":null},' \
        '"time_correction":null,"time_residual":null,' \
        '"takeoff_angle":null}]}'

cat = read_events(BytesIO(event_bytes), format='quakeml')

# ui_picks = json.loads(picks)



# @pytest.fixture
# def catalog():
#     event_bytes = requests.get(re.event_file).content
#     yield event_bytes
#
# @pytest.fixture
# def waveform_stream():
#     mseed_bytes = requests.get(re.waveform_file).content
#     yield mseed_bytes
#
# @pytest.fixture
# def picks():
#     picks = '{"event_resource_id":"smi:local/2019/06/27/08/25_47_115750099.e' \
#            '","data":[{"arrival_resource_id":"smi:local/ffbe62af-4b48-49ae' \
#             '-9be3-36e236a30ad5",' \
#             '"event":"smi:local/2019/06/27/08/25_47_115750099.e",' \
#             '"origin":"smi:local/0a49d7b8-1919-4ba6-aaaa-e835535ba0dd",' \
#             '"pick":{"pick_resource_id":"smi:local/84f3535f-ce76-4e0e-a048' \
#             '-49f637f18e58",' \
#             '"event":"smi:local/2019/06/27/08/25_47_115750099.e","site":1,' \
#             '"network":1,"station":37,' \
#             '"time_utc":"2019-06-27T08:25:48.021550Z","time_errors":null,' \
#             '"method_id":null,"filter_id":null,"onset":null,' \
#             '"phase_hint":"P","polarity":null,"evaluation_mode":"automatic",' \
#             '"evaluation_status":"preliminary"},"site":1,"network":1,' \
#             '"phase":"P","time_correction":null,"azimuth":37.0213943204641,' \
#             '"distance":1566.39623925138,"takeoff_angle":100.520098920753,' \
#             '"time_residual":-0.012353,' \
#             '"earth_model":"smi:local/f9faa611-fcd7-42d4-908d-e5b327e42bf6' \
#             '"},{"pick":{"site":1,"network":1,"station":78,' \
#             '"time_utc":"2019-06-27T08:25:47.889916Z","time_errors":null,' \
#             '"method_id":null,"filter_id":null,"onset":null,' \
#             '"phase_hint":"P","polarity":null,"evaluation_mode":"manual",' \
#             '"evaluation_status":null},"site":1,"network":1,"phase":"P",' \
#             '"time_correction":null,"azimuth":null,"distance":null,' \
#             '"takeoff_angle":null,"time_residual":null,"earth_model":null},' \
#             '{"arrival_resource_id":"smi:local/ff1ea391-a6ff-48ff-aee9' \
#             '-95ea70d3359a",' \
#             '"event":"smi:local/2019/06/27/08/25_47_115750099.e",' \
#             '"origin":"smi:local/0a49d7b8-1919-4ba6-aaaa-e835535ba0dd",' \
#             '"pick":{"pick_resource_id":"smi:local/3d185b2a-360f-44fa-9ed6' \
#             '-9e2be6e6e207",' \
#             '"event":"smi:local/2019/06/27/08/25_47_115750099.e","site":1,' \
#             '"network":1,"station":9,' \
#             '"time_utc":"2019-06-27T08:25:47.994092Z","time_errors":null,' \
#             '"method_id":null,"filter_id":null,"onset":null,' \
#             '"phase_hint":"P","polarity":null,"evaluation_mode":"automatic",' \
#             '"evaluation_status":"preliminary"},"site":1,"network":1,' \
#             '"phase":"P","time_correction":null,"azimuth":244.753316549404,' \
#             '"distance":1337.30104769653,"takeoff_angle":92.0077229506251,' \
#             '"time_residual":0.007648,' \
#             '"earth_model":"smi:local/f9faa611-fcd7-42d4-908d-e5b327e42bf6' \
#             '"},{"azimuth":null,"distance":null,"earth_model":null,' \
#             '"phase":"P","pick":{"evaluation_mode":"manual",' \
#             '"phase_hint":"P","station":"114",' \
#             '"time_utc":"2019-06-27T08:25:47.389011Z",' \
#             '"evaluation_status":null,"time_errors":null,"method_id":null,' \
#             '"filter_id":null,"onset":null,"polarity":null},' \
#             '"time_correction":null,"time_residual":null,' \
#             '"takeoff_angle":null},{"azimuth":null,"distance":null,' \
#             '"earth_model":null,"phase":"P","pick":{' \
#             '"evaluation_mode":"manual","phase_hint":"P","station":"119",' \
#             '"time_utc":"2019-06-27T08:25:47.381685Z",' \
#             '"evaluation_status":null,"time_errors":null,"method_id":null,' \
#             '"filter_id":null,"onset":null,"polarity":null},' \
#             '"time_correction":null,"time_residual":null,' \
#             '"takeoff_angle":null},{"azimuth":null,"distance":null,' \
#             '"earth_model":null,"phase":"S","pick":{' \
#             '"evaluation_mode":"manual","phase_hint":"S","station":"41",' \
#             '"time_utc":"2019-06-27T08:25:47.500733Z",' \
#             '"evaluation_status":null,"time_errors":null,"method_id":null,' \
#             '"filter_id":null,"onset":null,"polarity":null},' \
#             '"time_correction":null,"time_residual":null,' \
#             '"takeoff_angle":null}]}'
#
#     yield json.loads(picks)
#
#

waveform_stream = mseed_bytes


def test_interactive_pipeline(picks, event_bytes, mseed_bytes):
    params = {'waveform_bytes': mseed_bytes,
              'event_bytes'   : event_bytes,
              'picks_jsonb'         : picks}

    msg = msgpack.dumps(params)

    from microquake.db.connectors import connect_redis

    redis = connect_redis()

    message_queue = settings.get('processing_flow').interactive.message_queue

    # logger.info('sending request to the interactive pipeline on channel %s'
    #             % message_queue)

    redis.rpush(message_queue, msg)



test_interactive_pipeline(picks, event_bytes, mseed_bytes)


    # result = interactive_pipeline(**params)
