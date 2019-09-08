import json
from io import BytesIO

import numpy as np
from dateutil.parser import parse
from loguru import logger
from obspy import read_events, UTCDateTime
from obspy.core.event import CreationInfo, ResourceIdentifier, WaveformStreamID

from microquake.core import read
from microquake.core.event import Arrival, Origin, Pick, Magnitude
from microquake.core.settings import settings
from microquake.processors import (focal_mechanism, magnitude,
                                   measure_amplitudes, measure_energy,
                                   measure_smom, nlloc, magnitude_extractor,
                                   quick_magnitude)


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

    cat = prepare_catalog(picks, cat)

    # find traces with nans, which will choke `detrend()` calls:
    trs_with_nan = [tr for tr in stream.traces if np.isnan(tr.data.max())]
    for tr in trs_with_nan:
        logger.warning(f"Found nan in stream for {tr.id}")
    stream.traces = [tr for tr in stream.traces if tr not in trs_with_nan]

    # TODO this looks horrible, I shouldn't need to do this!
    for tr in stream:
        tr.stats.network = settings.SITE_CODE

    nlloc_processor = nlloc.Processor()
    cat_nlloc = nlloc_processor.process(cat=cat)['cat']

    # Removing the Origin object used to hold the picks
    del cat_nlloc[0].origins[-2]

    quick_magnitude_processor = quick_magnitude.Processor()
    quick_magnitude_processor.process(stream=stream, cat=cat_nlloc)
    cat_qm = quick_magnitude_processor.output_catalog(cat_nlloc)

    bytes_out = BytesIO()
    cat_nlloc.write(bytes_out, format='QUAKEML')

    m_amp_processor = measure_amplitudes.Processor()
    cat_amplitude = m_amp_processor.process(cat=cat_qm,
                                            stream=stream)['cat']

    smom_processor = measure_smom.Processor()
    cat_smom = smom_processor.process(cat=cat_amplitude,
                                      stream=stream)['cat']

    fmec_processor = focal_mechanism.Processor()
    cat_fmec = fmec_processor.process(cat=cat_smom,
                                      stream=stream)['cat']

    energy_processor = measure_energy.Processor()
    cat_energy = energy_processor.process(cat=cat_fmec,
                                          stream=stream)['cat']

    magnitude_processor = magnitude.Processor()
    cat_magnitude = magnitude_processor.process(cat=cat_energy,
                                                stream=stream)['cat']

    magnitude_f_processor = magnitude.Processor(module_type='frequency')
    cat_magnitude_f = magnitude_f_processor.process(cat=cat_magnitude,
                                                    stream=stream)['cat']

    mag = magnitude_extractor.Processor().process(cat=cat_magnitude_f)
    # send the magnitude info to the API

    origin_id = cat_magnitude_f[0].preferred_origin().resource_id
    corner_frequency = cat_magnitude_f[0].preferred_origin().comments[0]
    mag_obj = Magnitude(mag=mag['moment_magnitude'], magnitude_type='Mw',
                        origin_id=origin_id, evaluation_mode='automatic',
                        evaluation_status='preliminary',
                        comments=[corner_frequency])

    cat_magnitude_f[0].magnitudes.append(mag_obj)
    cat_magnitude_f[0].preferred_magnitude_id = mag_obj.resource_id

    return cat_magnitude_f, mag

