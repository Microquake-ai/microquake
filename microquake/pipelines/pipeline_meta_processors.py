from time import time

from loguru import logger
import numpy as np
from microquake.core.event import Magnitude
from microquake.processors import (focal_mechanism, magnitude,
                                   measure_amplitudes, measure_energy,
                                   measure_smom, nlloc, picker,
                                   magnitude_extractor, ray_tracer)


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


def picking_meta_processor(cat, fixed_length):

    logger.info('starting picking process')
    start_processing_time = time()

    loc = cat[0].preferred_origin().loc
    event_time_utc = cat[0].preferred_origin().time

    # cat_pe, picker_type = picker_election(loc, event_time_utc, cat,
    #                                      fixed_length)

    picker_p_processor = picker.Processor(module_type='high_frequency')
    picker_p_processor.process(stream=fixed_length, cat=cat, phase_filter='P')

    cat_picker_p = picker_p_processor.output_catalog(cat.copy())

    nlloc_processor = nlloc.Processor()
    nlloc_processor.initializer()
    cat_nlloc = nlloc_processor.process(cat=cat_picker_p)['cat']

    picker_ps_processor = picker.Processor(module_type='second_pass')
    picker_ps_processor.process(stream=fixed_length, cat=cat_nlloc)

    end_processing_time = time()

    processing_time = end_processing_time - start_processing_time

    logger.info(f'done picking in {processing_time} seconds')

    return picker_ps_processor.output_catalog(cat)


def location_meta_processor(cat):

    logger.info('starting location process')
    start_processing_time = time()

    nlloc_processor = nlloc.Processor()
    nlloc_processor.initializer()
    cat_nlloc = nlloc_processor.process(cat=cat)['cat']

    end_processing_time = time()
    processing_time = end_processing_time - start_processing_time
    logger.info(f'done locating event in {processing_time} seconds')

    logger.info('calculating rays')
    rt_start_time = time()
    rtp = ray_tracer.Processor()
    rtp.process(cat=cat_nlloc)
    cat_ray_tracer = rtp.output_catalog(cat_nlloc)
    rt_end_time = time()
    rt_processing_time = rt_end_time - rt_start_time
    logger.info(f'done calculating rays in {rt_processing_time} seconds')

    return cat_ray_tracer


def magnitude_meta_processor(cat, fixed_length):

    logger.info('starting magnitude calculation process')
    start_processing_time = time()

    m_amp_processor = measure_amplitudes.Processor()
    cat_amplitude = m_amp_processor.process(cat=cat,
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

    mag = magnitude_extractor.Processor().process(cat=cat_magnitude_f,
                                                  stream=fixed_length)

    cat_out = cat_magnitude_f.copy()
    preferred_origin_id = cat_magnitude_f[0].preferred_origin().resource_id
    new_mag = Magnitude.from_dict(mag, origin_id=preferred_origin_id)

    cat_out[0].magnitudes.append(new_mag)
    cat_out[0].preferred_magnitude_id = new_mag.resource_id

    end_processing_time = time()
    processing_time = end_processing_time - start_processing_time
    logger.info(f'done calculating magnitude in {processing_time} seconds')

    return cat_out, mag
