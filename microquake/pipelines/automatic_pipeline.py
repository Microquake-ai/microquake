from time import time

from loguru import logger

from microquake.core import Stream
from microquake.core.event import Catalog
from microquake.core.settings import settings
from microquake.pipelines.pipeline_meta_processors import (
    location_meta_processor,
    picking_meta_processor,
    ray_tracer,
)
from microquake.processors import clean_data, simple_magnitude


def automatic_pipeline(cat: Catalog, stream: Stream):
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
        rtp.process(cat=cat)
        cat_ray = rtp.output_catalog(cat)
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
        rtp.process(cat=cat)
        cat_ray = rtp.output_catalog(cat)
        return cat_ray

    cat_magnitude = simple_magnitude.Processor().process(cat=cat_located,
                                                         stream=stream)

    return cat_magnitude


def automatic_pipeline_test(cat: Catalog, stream: Stream):

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
