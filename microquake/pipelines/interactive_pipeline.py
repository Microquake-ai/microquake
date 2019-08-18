import json
from io import BytesIO

from dateutil.parser import parse
from obspy import read, read_events
from obspy.core.event import CreationInfo, ResourceIdentifier, WaveformStreamID

from microquake.core.event import Arrival, Origin, Pick
from microquake.core.settings import settings
from microquake.processors import (event_database, focal_mechanism, magnitude, measure_amplitudes, measure_energy,
                                   measure_smom, nlloc)


def prepare_catalog(ui_picks, catalog):
    """
    Takes the picks returned by the waveform UI and populate the catalog
    object to be located using NLLOC.
    :param picks:
    :param catalog:
    :return:
    """
    cat = catalog
    new_origin = Origin(x=0, y=0, z=0, time=UTCDateTime(),
                        evaluation_mode='manual',
                        evaluation_status='preliminary')
    new_origin.creation_info = CreationInfo(creation_time=UTCDateTime.now())
    new_origin.method_id = ResourceIdentifier("PICKER_FOR_HOLDING_ARRIVALS")

    for arrival in ui_picks['data']:
        for key in arrival.keys():
            if key == 'pick':
                # Determine if a pick needs to be appended to the pick list
                temp_pick = arrival['pick']
                date_time = UTCDateTime(parse(temp_pick['time_utc']))
                temp_pick['time'] = UTCDateTime(date_time)
                waveform_id = WaveformStreamID(
                    network_code=settings.NETWORK_CODE,
                    station_code=temp_pick['station'])

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
                            if temp_pick['time'] == pk_cat.time or temp_pick[
                                    'phase_hint'] == pk_cat.phase_hint:
                                # do not create a new pick
                                new_arrival = Arrival(phase=arrival['phase'],
                                                      pick_id=pk_cat.resource_id)
                            else:
                                new_pick = pk_cat.copy()
                                new_pick.resource_id = ResourceIdentifier()
                                new_pick.time = temp_pick['time']
                                new_pick.phase_hint = temp_pick['phase_hint']
                                new_arrival = Arrival(phase=temp_pick[
                                    'phase_hint', ])

                            new_origin.arrivals.append(new_arrival)

    cat[0].origins.append(new_origin)
    cat[0].preferred_origin_id = new_origin.resource_id.id

    return cat


def interactive_pipeline(waveform_bytes=None,
                         event_bytes=None,
                         picks_jsonb=None):
    """
    manual or interactive pipeline
    :param stream_bytes:
    :param cat_bytes:
    :return:
    """

    stream = read(BytesIO(waveform_bytes), format='mseed')
    cat = read_events(BytesIO(event_bytes), format='quakeml')
    picks = dict(json.loads(picks_jsonb))

    cat = prepare_catalog(picks, cat)

    eventdb_processor = event_database.Processor()
    eventdb_processor.initializer()

    # Error in postion data to the API. Returned with error code 400: bad
    # request

    nlloc_processor = nlloc.Processor()
    nlloc_processor.initializer()
    cat_nlloc = nlloc_processor.process(cat=cat)['cat']

    # Send the NLLOC result to the database
    result = eventdb_processor.process(cat=cat_nlloc)

    # Removing the Origin object used to hold the picks
    del cat_nlloc[0].origins[-2]

    # calculating the rays asynchronously
    bytes_out = BytesIO()
    cat_nlloc.write(bytes_out, format='QUAKEML')

    measure_amplitudes_processor = measure_amplitudes.Processor()
    cat_amplitude = measure_amplitudes_processor.process(cat=cat_nlloc,
                                                         stream=stream)['cat']

    smom_processor = measure_smom.Processor()
    cat_smom = smom_processor.process(cat=cat_amplitude,
                                      stream=stream)['cat']

    # TESTED UP TO THIS POINT, THE CONTAINER DOES NOT CONTAIN THE MOST
    # RECENT VERSION OF THE HASHWRAPPER LIBRARY AND CANNOT RUN
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

    # result = eventdb_processor.process(cat=cat_magnitude_f)

    return cat_magnitude_f
