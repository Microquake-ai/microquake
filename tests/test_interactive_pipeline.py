# import pytest
# from tests.helpers.data_utils import get_test_data

from datetime import timedelta
from io import BytesIO

import msgpack
import obspy
import pytest
import requests

from cachier import cachier
from microquake.clients import api_client
from microquake.core import read, read_events
from microquake.core.settings import settings
from microquake.db.models.redis import get_event, set_event

# from spp.pipeline.interactive_pipeline import interactive_pipeline

event_id = "smi:local/2019/06/27/08/25_47_115750099.e"


@pytest.fixture(scope="module", autouse=True)
@cachier(stale_after=timedelta(days=3))
def api_event(api_url):
    re = api_client.get_event_by_id(api_url, event_id)

    event_bytes = requests.get(re.event_file).content
    mseed_bytes = requests.get(re.waveform_file).content
    # 'catalog': read_events(BytesIO(event_bytes), format='quakeml'),

    res = {'waveform_bytes': mseed_bytes,
           'event_bytes': event_bytes}

    return res


def test_rom(api_event, picks):
    catalog = read_events(BytesIO(api_event['event_bytes']), format='quakeml')
    waveform = read(BytesIO(api_event['waveform_bytes']), format='mseed')

    set_event(event_id, catalogue=catalog, fixed_length=waveform)
    event = get_event(event_id)
    assert isinstance(event['catalogue'], obspy.core.event.catalog.Catalog)


@pytest.mark.skip(reason="this is only for testing purposes")
def test_put_event(api_url, api_event, picks):
    catalog = read_events(BytesIO(api_event['event_bytes']), format='quakeml')
    waveform = read(BytesIO(api_event['waveform_bytes']), format='mseed')
    network = 'OT'

    api_url = "http://localhost:8000/api/v1/"
    api_client.post_data_from_objects(api_url, network, event_id=None,
                                      cat=catalog,
                                      stream=waveform, tolerance=None,
                                      send_to_bus=False)
    event_id = "smi:local/2019/06/27/07/46_32_689969117.e"
    api_client.put_event_from_objects(api_url, network, event_id=event_id,
                                      event=catalog, waveform=waveform)


def test_interactive_pipeline(api_event, redis):
    msg = msgpack.dumps(api_event)

    message_queue = settings.get('processing_flow').interactive.message_queue

    redis.rpush(message_queue, msg)


# result = interactive_pipeline(**params)
# ui_picks = json.loads(picks)
