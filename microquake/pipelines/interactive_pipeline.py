import json
from io import BytesIO

import numpy as np
from dateutil.parser import parse
from loguru import logger
from microquake.core import read_events, read
from obspy import UTCDateTime
from obspy.core.event import (CreationInfo, ResourceIdentifier,
                              WaveformStreamID)

from microquake.pipelines.pipeline_meta_processors import \
    location_meta_processor

from microquake.core.event import Arrival, Origin, Pick
from microquake.core.settings import settings
from microquake.processors import simple_magnitude


def prepare_catalog(ui_picks, catalog):
    """
    Takes the picks returned by the waveform UI and populate the catalog
    object to be located using NLLOC.
    :param ui_picks:
    :param catalog:
    :return:
    """
    cat = catalog
    new_origin = Origin(x=0, y=0, z=0, time=UTCDateTime(),
                        evaluation_mode='manual',
                        evaluation_status='preliminary')
    new_origin.creation_info = CreationInfo(creation_time=UTCDateTime.now())
    new_origin.method_id = ResourceIdentifier("PICKER_FOR_HOLDING_ARRIVALS")

    for arrival in ui_picks:
        if 'pick' not in arrival.keys():
            continue

        # Determine if a pick needs to be appended to the pick list
        temp_pick = arrival['pick']
        date_time = UTCDateTime(parse(temp_pick['time_utc']))
        temp_pick['time'] = UTCDateTime(date_time)
        waveform_id = WaveformStreamID(
            network_code=settings.NETWORK_CODE,
            station_code=temp_pick['sensor'])
        # TODO microquake has no concept of ^ "sensor" elsewhere

        if 'pick_resource_id' not in arrival['pick'].keys():
            # create new pick and append the pick to the pick list
            new_pick = Pick(**temp_pick)
            cat[0].picks.append(new_pick)
            cat[0].picks[-1].waveform_id = waveform_id
            new_arrival = Arrival()
            new_arrival.phase = arrival['phase']
            new_arrival.pick_id = new_pick.resource_id
            new_origin.arrivals.append(new_arrival)

        else:
            for pk_cat in cat[0].picks:
                if temp_pick['pick_resource_id'] == pk_cat.resource_id:
                    if (temp_pick['time'] == pk_cat.time or
                            temp_pick['phase_hint'] == pk_cat.phase_hint):
                        # do not create a new pick
                        new_arrival = Arrival(phase=arrival['phase'],
                                              pick_id=pk_cat.resource_id)
                    else:
                        new_pick = pk_cat.copy()
                        new_pick.resource_id = ResourceIdentifier()
                        new_pick.time = temp_pick['time']
                        new_pick.phase_hint = temp_pick['phase_hint']
                        new_arrival = Arrival(phase=temp_pick['phase_hint'])

                    new_origin.arrivals.append(new_arrival)

    cat[0].origins.append(new_origin)
    cat[0].preferred_origin_id = new_origin.resource_id.id

    return cat


def interactive_pipeline(
        event_bytes: bytes, waveform_bytes: bytes, picks_jsonb: str):
    """
    manual or interactive pipeline
    :param waveform_bytes:
    :param event_bytes:
    :param picks_jsonb:
    :return:
    """

    cat = read_events(BytesIO(event_bytes), format='quakeml')
    stream = read(BytesIO(waveform_bytes), format='mseed')
    picks = json.loads(picks_jsonb)

    if cat[0].preferred_origin() is None:
        cat[0].preferred_origin_id = cat[0].origins[-1].resource_id

    cat = prepare_catalog(picks, cat)

    # find traces with nans, which will choke `detrend()` calls:
    trs_with_nan = [tr for tr in stream.traces if np.isnan(tr.data.max())]
    for tr in trs_with_nan:
        logger.warning(f"Found nan in stream for {tr.id}")
    stream.traces = [tr for tr in stream.traces if tr not in trs_with_nan]

    # TODO this looks horrible, I shouldn't need to do this!
    for tr in stream:
        tr.stats.network = settings.SITE_CODE

    cat_located = location_meta_processor(cat)

    # nlloc_processor = nlloc.Processor()
    # cat_nlloc = nlloc_processor.process(cat=cat)['cat']
    #
    # rt_processor = ray_tracer.Processor()
    # rt_processor.process(cat=cat_nlloc)
    # cat_rays = rt_processor.output_catalog(cat_nlloc)

    # Removing the Origin object used to hold the picks
    # try:
    cat_magnitude = simple_magnitude.Processor().process(cat=cat_located,
                                                             stream=stream)
    # except ValueError as ve:
    #     logger.error(f'Calculation of the magnitude failed. \n{ve}')
    #     cat_magnitude = cat_located.copy()

    cat_magnitude[0].preferred_origin().evaluation_mode = 'manual'
    cat_magnitude[0].preferred_origin().evaluation_status = 'final'

    return cat_magnitude

