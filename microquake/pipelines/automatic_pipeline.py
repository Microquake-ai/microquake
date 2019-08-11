from io import BytesIO

import numpy as np
import requests

from loguru import logger
from microquake.clients.api_client import put_event_from_objects, reject_event
from microquake.core.event import Catalog, Event
from microquake.core.settings import settings
from microquake.db.connectors import RedisQueue
from microquake.processors import (clean_data, focal_mechanism, magnitude, measure_amplitudes, measure_energy,
                                   measure_smom, nlloc, picker)
from spp.core.serializers.seismic_objects import deserialize_message, serialize

api_base_url = settings.get('api_base_url')

api_message_queue = settings.API_MESSAGE_QUEUE
api_queue = RedisQueue(api_message_queue)


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
    picker_hf_processor = picker.Processor(module_type='high_frequencies')
    picker_mf_processor = picker.Processor(module_type='medium_frequencies')
    picker_lf_processor = picker.Processor(module_type='low_frequencies')

    response_hf = picker_hf_processor.process(stream=stream, location=location,
                                              event_time_utc=event_time_utc)
    response_mf = picker_mf_processor.process(stream=stream, location=location,
                                              event_time_utc=event_time_utc)
    response_lf = picker_lf_processor.process(stream=stream, location=location,
                                              event_time_utc=event_time_utc)

    cat_pickers = []

    if response_hf:
        cat_picker_hf = picker_hf_processor.output_catalog(cat.copy())
        cat_pickers.append(cat_picker_hf)

    if response_mf:
        cat_picker_mf = picker_mf_processor.output_catalog(cat.copy())
        cat_pickers.append(cat_picker_mf)

    if response_lf:
        cat_picker_lf = picker_lf_processor.output_catalog(cat.copy())
        cat_pickers.append(cat_picker_lf)

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

    imax = np.argmax(len_arrivals)

    return cat_pickers[imax]


@deserialize_message
def put_data_api(catalogue=None, fixed_length=None, **kwargs):
    event_id = catalogue[0].resource_id.id

    response = put_event_from_objects(api_base_url, event_id,
                                      event=catalogue,
                                      waveform=fixed_length)

    if response.status_code != requests.codes.ok:
        logger.info('request failed, resending to the queue')
        dict_out = {'catalogue': catalogue,
                    'fixed_length': fixed_length}

        message = serialize(**dict_out)

        result = api_queue.submit_task(put_data_api,
                                       kwargs={'data': message,
                                               'serialized': True})

        return result


@deserialize_message
def automatic_pipeline(catalogue=None, fixed_length=None, **kwargs):
    """
    automatic pipeline
    :param fixed_length: fixed length stream encoded as mseed
    :param catalogue: catalog object encoded in quakeml
    :return:
    """

    stream = fixed_length

    if catalogue is None:
        logger.info('No catalog was provided creating new')
        cat = Catalog(events=[Event()])
    else:
        cat = catalogue

    event_id = cat[0].resource_id

    logger.info('removing traces for sensors in the black list, or are '
                'filled with zero, or contain NaN')
    clean_data_processor = clean_data.Processor()
    fixed_length = clean_data_processor.process(waveform=fixed_length)

    loc = cat[0].preferred_origin().loc
    event_time_utc = cat[0].preferred_origin().time

    cat_picker = picker_election(loc, event_time_utc, cat, stream)

    if not cat_picker:
        logger.warning('The picker did not return any picks! Marking the '
                       'event as rejected in the database.')

        api_queue.submit_task(reject_event,
                              args=(api_base_url,
                                    cat[0].resource_id.id))

        return False

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

    bytes_out = BytesIO()
    cat_nlloc.write(bytes_out, format='QUAKEML')

    # send to data base
    cat_nlloc[0].resource_id = event_id

    message = serialize(catalogue=cat_nlloc)

    api_queue.submit_task(put_data_api, kwargs={'data': message,
                                                'serialized': True})

    # put_event_from_objects(api_base_url, event_id, event=cat_nlloc,
    #                        waveform=fixed_length)

    measure_amplitudes_processor = measure_amplitudes.Processor()
    cat_amplitude = measure_amplitudes_processor.process(cat=cat_nlloc,
                                                         stream=fixed_length)['cat']

    smom_processor = measure_smom.Processor()
    cat_smom = smom_processor.process(cat=cat_amplitude,
                                      stream=fixed_length)['cat']

    fmec_processor = focal_mechanism.Processor()
    cat_fmec = fmec_processor.process(cat=cat_smom,
                                      stream=fixed_length)['cat']

    energy_processor = measure_energy.Processor()
    cat_energy = energy_processor.process(cat=cat_fmec,
                                          stream=fixed_length)['cat']

    magnitude_processor = magnitude.Processor()
    cat_magnitude = magnitude_processor.process(cat=cat_energy,
                                                stream=fixed_length)['cat']

    magnitude_f_processor = magnitude.Processor(module_type='frequency')
    cat_magnitude_f = magnitude_f_processor.process(cat=cat_magnitude,
                                                    stream=fixed_length)['cat']

    cat_magnitude_f[0].resource_id = event_id

    message = serialize(catalogue=cat_magnitude_f)
    api_queue.submit_task(put_data_api, kwargs={'data': message,
                                                'serialized': True})
    # put_event_from_objects(api_base_url, event_id, event=cat_magnitude_f)

    return cat_magnitude_f
