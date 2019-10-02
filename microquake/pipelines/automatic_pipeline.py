from io import BytesIO

import requests
from microquake.core.event import Catalog
from time import time

from loguru import logger
from microquake.core.event import Event
from microquake.clients.api_client import (post_data_from_objects,
                                           get_event_by_id)
from microquake.core.settings import settings
from microquake.db.connectors import RedisQueue, record_processing_logs_pg
from microquake.db.models.redis import get_event, set_event
from microquake.processors import clean_data, simple_magnitude
from microquake.pipelines.pipeline_meta_processors import (ray_tracer,
    picking_meta_processor, location_meta_processor)

__processing_step__ = 'automatic processing'
__processing_step_id__ = 3

api_base_url = settings.get('api_base_url')


def put_data(event_id, **kwargs):

    import requests

    api_message_queue = settings.API_MESSAGE_QUEUE
    api_queue = RedisQueue(api_message_queue)

    processing_step = 'update_event_api'
    processing_step_id = 5
    processing_start_time = time()
    event_key = event_id
    event = get_event(event_key)

    response = put_data_processor(event['catalogue'])
    logger.info(response.status_code)

    if 200 <= response.status_code < 400:

        logger.info('request failed, resending to the queue')

        result = api_queue.submit_task(put_data, event_id=event_key)

        processing_end_time = time()
        processing_time = processing_end_time - processing_start_time
        record_processing_logs_pg(event['catalogue'], 'success',
                                  processing_step, processing_step_id,
                                  processing_time)

    return response


def put_data_processor(catalog):

    from uuid import uuid4

    event_id = catalog[0].resource_id.id

    base_url = api_base_url
    if base_url[-1] == '/':
        base_url = base_url[:-1]

    # check if the event type and event status has changed on the API

    re = get_event_by_id(api_base_url, event_id)
    try:
        catalog[0].event_type = re.event_type
        catalog[0].preferred_origin().evaluation_status = re.status
    except AttributeError as e:
        logger.error(e)

    url = f'{base_url}/events/{event_id}/files'

    cat_bytes = BytesIO()
    catalog.write(cat_bytes)

    file_name = str(uuid4()) + '.xml'
    cat_bytes.name = file_name
    cat_bytes.seek(0)

    files = {'event': cat_bytes}

    logger.info(f'attempting to PUT catalog for event {event_id}')

    response = requests.put(url, files=files)

    logger.info(f'API responded with {response.status_code} code')
    return response


def automatic_pipeline(event_id, **kwargs):

    api_message_queue = settings.API_MESSAGE_QUEUE
    api_queue = RedisQueue(api_message_queue)

    start_processing_time = time()

    event = get_event(event_id)
    stream = event['fixed_length']

    if event['catalogue'] is None:
        logger.info('No catalog was provided creating new')
        cat = Catalog(events=[Event()])
    else:
        cat = event['catalogue']

    automatic_pipeline_processor(cat, stream)

    set_event(event_id, catalogue=cat_magnitude)
    api_queue.submit_task(put_data, event_id=event_id)

    end_processing_time = time()
    processing_time = end_processing_time - start_processing_time

    record_processing_logs_pg(event['catalogue'], 'success',
                              __processing_step__, __processing_step_id__,
                              processing_time)

    logger.info(f'automatic processing completed in {processing_time} seconds')

    return cat_magnitude


def post_event_api(event_id, **kwargs):

    api_message_queue = settings.API_MESSAGE_QUEUE
    api_queue = RedisQueue(api_message_queue)

    processing_step = 'post_event_api'
    processing_step_id = 4
    start_processing_time = time()

    event = get_event(event_id)
    response = post_data_from_objects(api_base_url, event_id=None,
                                      event=event['catalogue'],
                                      stream=event['fixed_length'],
                                      tolerance=None,
                                      send_to_bus=False)
    if 200 <= response.status_code < 400:
        logger.info('request failed, resending to the queue')
        result = api_queue.submit_task(post_event_api, event_id=event_id)
        return result


def automatic_pipeline_processor(cat, stream):
    logger.info('removing traces for sensors in the black list, or are '
                'filled with zero, or contain NaN')
    clean_data_processor = clean_data.Processor()
    fixed_length = clean_data_processor.process(waveform=stream)

    cat_picked = picking_meta_processor(cat, fixed_length)
    rtp = ray_tracer.Processor()

    min_number_pick = settings.get('picker').min_num_picks
    n_picks = len(cat_picked[0].preferred_origin().arrivals)
    if n_picks < min_number_pick:
        logger.warning(f'number of picks ({n_picks}) is lower than the '
                       f'minimum number of picks ({min_number_pick}). '
                       f'Aborting automatic processing!')
        if cat[0].preferred_origin().rays:
            return cat
        cat_ray = rtp.process(cat=cat)
        return cat_ray

    cat_located = location_meta_processor(cat_picked)

    max_uncertainty = settings.get('location').max_uncertainty
    uncertainty = cat_located[0].preferred_origin().uncertainty
    if uncertainty > max_uncertainty:
        logger.warning(f'uncertainty ({uncertainty} m) is above the '
                       f'threshold of {max_uncertainty} m. Aborting '
                       f'automatic processing!')
        if cat[0].preferred_origin().rays:
            return cat
        cat_ray = rtp.process(cat=cat)
        return cat_ray

    cat_magnitude = simple_magnitude.Processor().process(cat=cat_located,
                                                         stream=stream)

    return cat_magnitude


def automatic_pipeline_test(cat, stream):

    start_processing_time = time()

    logger.info('removing traces for sensors in the black list, or are '
                'filled with zero, or contain NaN')
    clean_data_processor = clean_data.Processor()
    fixed_length = clean_data_processor.process(waveform=stream)

    cat_picked = picking_meta_processor(cat, fixed_length)

    min_number_pick = settings.get('picker').min_num_picks
    if len(cat_picked[0].preferred_origin().arrivals) < min_number_pick:
        return cat

    cat_located = location_meta_processor(cat_picked)

    max_uncertainty = settings.get('location').max_uncertainty
    if cat_located[0].preferred_origin().uncertainty > max_uncertainty:
        return cat

    cat_magnitude = simple_magnitude.Processor().process(cat=cat_located,
                                                         stream=stream)

    end_processing_time = time()
    processing_time = end_processing_time - start_processing_time
    logger.info(f'done automatic pipeline in {processing_time} seconds')

    return cat_magnitude
