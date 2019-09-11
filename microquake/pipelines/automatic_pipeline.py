from io import BytesIO

import numpy as np
import requests
from microquake.core.event import Catalog
from time import time

from loguru import logger
from microquake.clients.api_client import put_event_from_objects, reject_event
from microquake.core.event import Event, Magnitude
from microquake.clients.api_client import post_data_from_objects
from microquake.core.settings import settings
from microquake.db.connectors import RedisQueue, record_processing_logs_pg
from microquake.db.models.redis import get_event, set_event
from microquake.processors import (clean_data, focal_mechanism, magnitude,
                                   measure_amplitudes, measure_energy,
                                   measure_smom, nlloc, picker,
                                   magnitude_extractor)

__processing_step__ = 'automatic processing'
__processing_step_id__ = 3

api_base_url = settings.get('api_base_url')


def picker_election(location, event_time_utc, cat, stream):
    """
    Calculates the picks using 1 method but different
    parameters and then retains the best set of picks. The function is
    current calling the picker sequentially processes should be spanned so
    the three different picker are running over three distinct threads.
    :param cat: Catalog
    :param fixed_length: fixed length seismogram
    :return: a Catalog containing the response of the picker that performed
    best according to some logic described in this function.
    """

    picker_types = ['high_frequencies',
                    'medium_frequencies',
                    'low_frequencies']

    picker_0_processor = picker.Processor(module_type=picker_types[0])
    picker_1_processor = picker.Processor(module_type=picker_types[1])
    picker_2_processor = picker.Processor(module_type=picker_types[2])

    response_0 = picker_0_processor.process(stream=stream, location=location,
                                            event_time_utc=event_time_utc)
    response_1 = picker_1_processor.process(stream=stream, location=location,
                                            event_time_utc=event_time_utc)
    response_2 = picker_2_processor.process(stream=stream, location=location,
                                            event_time_utc=event_time_utc)

    cat_pickers = []

    if response_0:
        cat_picker_0 = picker_0_processor.output_catalog(cat.copy())
        cat_pickers.append(cat_picker_0)

    if response_1:
        cat_picker_1 = picker_1_processor.output_catalog(cat.copy())
        cat_pickers.append(cat_picker_1)

    if response_2:
        cat_picker_2 = picker_2_processor.output_catalog(cat.copy())
        cat_pickers.append(cat_picker_2)

    if not cat_pickers:
        return False

    len_arrivals = [len(catalog[0].preferred_origin().arrivals)
                    for catalog in cat_pickers]

    logger.info('Number of arrivals for each picker:\n'
                'High Frequencies picker   : %d \n'
                'Medium Frequencies picker : %d \n'
                'Low Frequencies picker    : %d \n' % (len_arrivals[0],
                                                       len_arrivals[1],
                                                       len_arrivals[2]))

    i_max = np.argmax(len_arrivals)

    return cat_pickers[i_max], picker_types[i_max]


def put_data_api(event_id, **kwargs):

    import requests

    api_message_queue = settings.API_MESSAGE_QUEUE
    api_queue = RedisQueue(api_message_queue)

    processing_step = 'update_event_api'
    processing_step_id = 5
    processing_start_time = time()
    event_key = event_id
    event = get_event(event_key)

    response = put_data_processor(event['catalogue'])

    if response.status_code != requests.codes.ok:

        logger.info('request failed, resending to the queue')

        result = api_queue.submit_task(put_data_api, event_id=event_key)

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

    url = f'{base_url}/events/{event_id}/files'

    cat_bytes = BytesIO()
    catalog.write(cat_bytes)

    file_name = str(uuid4()) + '.xml'
    cat_bytes.name = file_name
    cat_bytes.seek(0)

    files = {'event': cat_bytes}

    response = requests.put(url, files=files)
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

    cat_out, mag = automatic_processor(cat, stream)

    set_event(event_id, catalogue=cat)
    api_queue.submit_task(put_data_api, event_id=event_id)

    end_processing_time = time()
    processing_time = end_processing_time - start_processing_time

    record_processing_logs_pg(event['catalogue'], 'success',
                              __processing_step__, __processing_step_id__,
                              processing_time)

    return cat_out, mag


def automatic_pipeline_api(event_id, **kwargs):

    start_processing_time = time()

    event = get_event(event_id)
    stream = event['fixed_length']

    if event['catalogue'] is None:
        logger.info('No catalog was provided creating new')
        cat = Catalog(events=[Event()])
    else:
        cat = event['catalogue']

    cat_out, mag = automatic_processor(cat, stream)

    return cat_out, mag


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

    if response.status_code != requests.codes.ok:
        logger.info('request failed, resending to the queue')

        result = api_queue.submit_task(post_event_api, event_id=event_id)


        return result


def automatic_processor(cat, stream):

    start_processing_time = time()

    logger.info('removing traces for sensors in the black list, or are '
                'filled with zero, or contain NaN')
    clean_data_processor = clean_data.Processor()
    fixed_length = clean_data_processor.process(waveform=stream)

    loc = cat[0].preferred_origin().loc
    event_time_utc = cat[0].preferred_origin().time

    cat_picker, picker_type = picker_election(loc, event_time_utc, cat,
                                              fixed_length)

    if not cat_picker:
        return

    nlloc_processor = nlloc.Processor()
    nlloc_processor.initializer()
    cat_nlloc = nlloc_processor.process(cat=cat_picker)['cat']

    # Removing the Origin object used to hold the picks
    del cat_nlloc[0].origins[-2]

    loc = cat_nlloc[0].preferred_origin().loc
    event_time_utc = cat_nlloc[0].preferred_origin().time
    picker_sp_processor = picker.Processor(module_type='second_pass')
    response = picker_sp_processor.process(stream=fixed_length, location=loc,
                                           event_time_utc=event_time_utc)

    if response is False:
        logger.warning('Picker failed aborting automatic processing!')

        return False

    cat_picker = picker_sp_processor.output_catalog(cat_nlloc)

    nlloc_processor = nlloc.Processor()
    nlloc_processor.initializer()
    cat_nlloc = nlloc_processor.process(cat=cat_picker)['cat']

    # Removing the Origin object used to hold the picks
    del cat_nlloc[0].origins[-2]

    m_amp_processor = measure_amplitudes.Processor()
    cat_amplitude = m_amp_processor.process(cat=cat_nlloc,
                                            stream=fixed_length)

    smom_processor = measure_smom.Processor()
    cat_smom = smom_processor.process(cat=cat_amplitude,
                                      stream=fixed_length)

    fmec_processor = focal_mechanism.Processor()
    cat_fmec = fmec_processor.process(cat=cat_smom,
                                      stream=fixed_length)

    energy_processor = measure_energy.Processor()
    cat_energy = energy_processor.process(cat=cat_fmec,
                                          stream=fixed_length)

    magnitude_processor = magnitude.Processor()
    cat_magnitude = magnitude_processor.process(cat=cat_energy,
                                                stream=fixed_length)

    magnitude_f_processor = magnitude.Processor(module_type='frequency')
    cat_magnitude_f = magnitude_f_processor.process(cat=cat_magnitude,
                                                    stream=fixed_length)

    end_processing_time = time()

    processing_time = end_processing_time - start_processing_time

    mag = magnitude_extractor.Processor().process(cat=cat_magnitude_f)
    # send the magnitude info to the API

    mag = magnitude_extractor.Processor().process(cat=cat_magnitude_f)

    cat_out = cat_magnitude_f.copy()
    preferred_origin_id = cat_magnitude_f[0].preferred_origin().resource_id
    new_mag = Magnitude.from_dict(mag, origin_id=preferred_origin_id)

    cat_out[0].magnitudes.append(new_mag)
    cat_out[0].preferred_magnitude_id = new_mag.resource_id

    return cat_out, mag
