import json
from io import BytesIO

from dateutil.parser import parse
from obspy import read_events, UTCDateTime
from obspy.core.event import CreationInfo, ResourceIdentifier, WaveformStreamID

from microquake.core import read
from microquake.core.event import Arrival, Origin, Pick
from microquake.core.settings import settings
from microquake.processors import (
    focal_mechanism, magnitude, measure_amplitudes, measure_energy,
    measure_smom, nlloc)


def prepare_catalog(ui_picks, catalog):
    """
    Takes the picks returned by the waveform UI and populate the catalog
    object to be located using NLLOC.
    :param picks:
    :param catalog:
    :return:
    """
    assert len(catalog) == 1
    cat = catalog
    new_origin = Origin(x=0, y=0, z=0, time=UTCDateTime(),
                        evaluation_mode='manual',
                        evaluation_status='preliminary')
    # TODO when do we change "preliminary"  ^  into something else?
    new_origin.creation_info = CreationInfo(creation_time=UTCDateTime.now())
    new_origin.method_id = ResourceIdentifier("PICKER_FOR_HOLDING_ARRIVALS")

    for i, existing_pick in enumerate(cat[0].picks):
        cat[0].picks[i] = Pick(existing_pick)  # obspy to microquake
        new_arrival = Arrival(
            phase=existing_pick.phase_hint, pick_id=existing_pick.resource_id)
        new_origin.arrivals.append(new_arrival)
        # TODO skip picks that match (station, phase), not time not, to one below

    for arrival in ui_picks:
        temp_pick = arrival['pick']
        date_time = UTCDateTime(parse(temp_pick['time_utc']))
        temp_pick['time'] = UTCDateTime(date_time)
        waveform_id = WaveformStreamID(
            network_code=settings.NETWORK_CODE,
            station_code=temp_pick['sensor'])
        # TODO microquake has no concept of "sensor" anywhere else ^

        # create new pick and append the pick to the pick list
        new_pick = Pick(**temp_pick)
        new_pick.waveform_id = waveform_id
        # TODO I have no idea why station_code is None without this line:
        new_pick.waveform_id.station_code = waveform_id.station_code
        cat[0].picks.append(new_pick)
        new_arrival = Arrival()
        new_arrival.phase = arrival['phase']
        new_arrival.pick_id = new_pick.resource_id
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

    # TODO this looks horrible, I shouldn't need to do this!
    for tr in stream:
        tr.stats.network = "HNUG"

    nlloc_processor = nlloc.Processor()
    cat_nlloc = nlloc_processor.process(cat=cat)['cat']

    # Removing the Origin object used to hold the picks
    del cat_nlloc[0].origins[-2]

    bytes_out = BytesIO()
    cat_nlloc.write(bytes_out, format='QUAKEML')

    measure_amplitudes_processor = measure_amplitudes.Processor()
    cat_amplitude = measure_amplitudes_processor.process(
        cat=cat_nlloc, stream=stream)['cat']

    smom_processor = measure_smom.Processor()
    cat_smom = smom_processor.process(
        cat=cat_amplitude, stream=stream)['cat']

    fmec_processor = focal_mechanism.Processor()
    cat_fmec = fmec_processor.process(
        cat=cat_smom, stream=stream)['cat']

    energy_processor = measure_energy.Processor()
    cat_energy = energy_processor.process(
        cat=cat_fmec, stream=stream)['cat']

    magnitude_processor = magnitude.Processor()
    cat_magnitude = magnitude_processor.process(
        cat=cat_energy, stream=stream)['cat']

    magnitude_f_processor = magnitude.Processor(module_type='frequency')
    cat_magnitude_f = magnitude_f_processor.process(
        cat=cat_magnitude, stream=stream)['cat']

    return cat_magnitude_f
