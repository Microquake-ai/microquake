import os

import numpy as np
from obspy.core.inventory.response import InstrumentSensitivity, PolesZerosResponseStage
from obspy.core.inventory.util import Frequency

import instResp
from instResp.libInst import get_corner_freq_from_pole, getResponse
from instResp.libNom import RC, WA, Accelerometer
from instResp.plotResp import plotResponse
from loguru import logger
from microquake.core.data.inventory import (Inventory, load_inventory, load_inventory_from_excel,
                                            test_print_OT_xml_summary)
from microquake.core.data.response_utils import read_cable_file, read_sensor_types_file

ns_tag = 'mq'
ns = 'MICROQUAKE'


def get_sensitivity(resistance):
    if resistance == 0:
        # print("    This is a most likely an Accelerometer!")
        sensitivity = 1.0
    elif resistance % 3500 == 0:
        # print("    This is a high-gain geophone --> Use 80 V/m/s")
        sensitivity = resistance/3500 * 80
    elif resistance % 375 == 0:
        # print("    This is a low-gain geophone --> Use 28.8 V/m/s")
        sensitivity = resistance/375 * 28.8
    else:
        # print("    Unknown resistance [%s] --> use default sensitivity=1.0" % resistance)
        sensitivity = 1.0

    return sensitivity


def fix_OT_responses(inventory):
    '''
    Replace the generic (placeholder) channel responses in inventory
        with calculated responses.

    Response calculations are made using values in the station extras dict
        like damping, coil_resistance, etc.
    '''

    for station in inventory.stations():

        logger.info("sta:%s sensor_id:%s" % (station.code, station.sensor_id))

        extras = station.extra
        resistance = extras.coil_resistance.value
        f0 = extras.resonance_frequency.value
        damp = extras.damping.value
        cable_cap = extras.cable_capacitance_pF_per_meter.value
        cable_len = extras.cable_length.value

        sensitivity = get_sensitivity(resistance)
        extras['calculated_pz_sensitivity'] = {'namespace': ns, 'value': sensitivity}

        logger.info("resistance:%f sensitivity:%f cable_cap:%f len:%f" %
                    (resistance, sensitivity, cable_cap, cable_len))

        pz_generator = WA
        input_units = "M/S"

        if station.motion == "ACCELERATION":
            pz_generator = Accelerometer
            input_units = "M/S**2"

        pzs = pz_generator(per=1/f0, damp=damp, gain=1.0, normalize=True, normalize_freq=100.)
        # MTH: JP wants sensitivity set to 1.0 since OT data already scaled to velocity/accel:
        pzs.sensitivity = 1.0
        pzs.sensitivity_f = 100.

        freqs = np.logspace(-5, 4., num=2000)

        # pzs.name = station.sensor_id
        pzs.name = extras.model.value
        pzs.unitsIn = input_units
        pzs.unitsOut = "V"

        if cable_cap == 0:
            # print("No cable cap set --> Skip!")
            pass
        else:
            # Cable capacity in pF (=10^-12 Farads):
            cable_capacity = cable_cap * 1e-12 * cable_len
            tau = resistance * cable_capacity
            f_rc = 1./tau
            # print("cap_per_m:%s x len:%f = %f  x R=%f --> tau=%f fc=1/tau=%g" % \
            # (cable_cap, cable_len, cable_capacity, resistance, tau, f_rc))
            pz_rc = RC(tau=tau)
            pzs.append_pole(pz_rc.poles[0])
            pzs.normalize_to_a0(norm_freq=100)

        resp = getResponse(pzs, freqs, removeZero=False, useSensitivity=False)

        title = 'sta:%s sensor_type:%s f0=%.0f Hz h=%.2f sensitivity=%.2f' % \
            (station.code, station.sensor_id, f0, damp, sensitivity)
        logger.info("Corner freq:%f" % get_corner_freq_from_pole(pzs.poles[0]))

        fc_low = -999.

        if station.motion == "VELOCITY":
            fc_low = get_corner_freq_from_pole(pzs.poles[0])
        # elif station.motion == "ACCELERATION":

        fc_high = 1e6

        if pzs.poles.size == 3:
            logger.info("** High-f Corner freq:%f" % get_corner_freq_from_pole(pzs.poles[2]))
            fc_high = get_corner_freq_from_pole(pzs.poles[2])

        extras['min_frequency'] = {'namespace': ns, 'value': float(fc_low)}
        extras['max_frequency'] = {'namespace': ns, 'value': float(fc_high)}

        # if station.code == '2':
        # if 1:
        # plotResponse(resp, freqs, title=title, xmin=1, xmax=10000., ymin=.01, ymax=6, title_font_size=8)
        # exit()

        response = station.channels[0].response
        instrument_sensitivity = response.instrument_sensitivity
        instrument_sensitivity.value = 1.
        instrument_sensitivity.frequency = 100.

        stages = response.response_stages
        # Insert OT geophone or accelerometer response in first stage of response:
        stages[0] = convert_pz_to_obspy(pzs)
        # Use generic digitizer for stage 2 with output sample rate = 6KHz
        stages[2].name = "Generic Digitizer = Placeholder for IMS Digitizer"
        stages[2].stage_gain = 1
        stages[2].decimation_input_sample_rate = Frequency(12000.)
        stages[2].decimation_factor = 2

        response.response_stages = stages[0:3]

        for channel in station.channels:
            channel.response = response

    return 1


def write_OT_xml(sensor_file, sensor_type_file, cable_file, xml_outfile='OT.xml'):
    '''
    Deprecated - used when network metadata was spread over individual csv files
    '''

    print("write_OT_xml: xml_outfile=%s" % xml_outfile)

    cables = read_cable_file(cable_file)
    sensor_types = read_sensor_types_file(sensor_type_file)
    inventory = load_inventory(sensor_file)

    for cable in cables:
        logger.info("cable:%s --> %s" % (cable, cables[cable]))

    for sensor in sensor_types:
        logger.info("sensor:%s --> %s" % (sensor, sensor_types[sensor]))

    for station in inventory.stations():

        logger.info("sta:%s sensor_id:%s" % (station.code, station.sensor_id))
        logger.info(sensor_types)

        sensor_type = sensor_types[station.sensor_id]
        cable_cap = cables[station.cable_type]

        extras = station.extra
        extras['output_resistance_ohm'] = {'namespace': ns, 'value': float(sensor_type['output resistance (ohm)'])}
        extras['resonance_frequency'] = {'namespace': ns, 'value': sensor_type['resonance frequency (Hz)']}
        extras['damping'] = {'namespace': ns, 'value': float(sensor_type['damping'])}
        extras['cable_pF_capacitance_per_m'] = {'namespace': ns, 'value': float(cable_cap)}

        resistance = float(sensor_type['output resistance (ohm)'])

        if resistance == 0:
            # print("    This is a most likely an Accelerometer!")
            sensitivity = 1.0
        elif resistance % 3500 == 0:
            # print("    This is a high-gain geophone --> Use 80 V/m/s")
            sensitivity = resistance/3500 * 80
        elif resistance % 375 == 0:
            # print("    This is a low-gain geophone --> Use 28.8 V/m/s")
            sensitivity = resistance/375 * 28.8
        else:
            # print("    Unknown resistance [%s] --> use default sensitivity=1.0" % resistance)
            sensitivity = 1.0

        pz_generator = WA
        input_units = "M/S"

        if station.motion == "ACCELERATION":
            pz_generator = Accelerometer
            input_units = "M/S**2"

        f0 = float(sensor_type['resonance frequency (Hz)'])
        damp = float(sensor_type['damping'])

        pzs = pz_generator(per=1/f0, damp=damp, gain=1.0, normalize=True, normalize_freq=100.)
        # MTH: JP wants sensitivity set to 1.0 since OT data already scaled to velocity/accel:
        pzs.sensitivity = 1.0
        pzs.sensitivity_f = 100.

        extras['calculated_pz_sensitivity'] = {'namespace': ns, 'value': float(sensitivity)}
        freqs = np.logspace(-5, 4., num=2000)

        pzs.name = station.sensor_id
        pzs.unitsIn = input_units
        pzs.unitsOut = "V"

        if cable_cap == "0":
            # print("No cable cap set --> Skip!")
            # print(pzs)
            pass
        else:
            cable_len = float(station.cable_length)
            # Cable capacity in pF (=10-12 Farads):
            cable_capacity = float(cable_cap) * 1e-12 * cable_len
            tau = resistance * cable_capacity
            f_rc = 1./tau
            # print("cap_per_m:%s x len:%f = %f  x R=%f --> tau=%f fc=1/tau=%g" % \
            # (cable_cap, cable_len, cable_capacity, resistance, tau, f_rc))
            pz_rc = RC(tau=tau)
            pzs.append_pole(pz_rc.poles[0])
            pzs.normalize_to_a0(norm_freq=100)

        resp = getResponse(pzs, freqs, removeZero=False, useSensitivity=False)

        title = 'sta:%s sensor_type:%s f0=%.0f Hz h=%.2f sensitivity=%.2f' % \
            (station.code, station.sensor_id, f0, damp, sensitivity)
        logger.info("Corner freq:%f" % get_corner_freq_from_pole(pzs.poles[0]))

        fc_low = -999.

        if station.motion == "VELOCITY":
            fc_low = get_corner_freq_from_pole(pzs.poles[0])
        # elif station.motion == "ACCELERATION":

        fc_high = 1e6

        if pzs.poles.size == 3:
            logger.info("** High-f Corner freq:%f" % get_corner_freq_from_pole(pzs.poles[2]))
            fc_high = get_corner_freq_from_pole(pzs.poles[2])

        extras['min_frequency'] = {'namespace': ns, 'value': float(fc_low)}
        extras['max_frequency'] = {'namespace': ns, 'value': float(fc_high)}

        # if station.code == '2':
        # if 1:
        # plotResponse(resp, freqs, title=title, xmin=1, xmax=10000., ymin=.01, ymax=6, title_font_size=8)
        # exit()

        from obspy.core.inventory.response import InstrumentSensitivity
        from obspy.core.inventory.util import Frequency
        """
        :type instrument_sensitivity:
            :class:`~obspy.core.inventory.response.InstrumentSensitivity`
        :param instrument_sensitivity: The total sensitivity for the given
            channel, representing the complete acquisition system expressed as
            a scalar.
           def __init__(self, value, frequency, input_units,
            output_units, input_units_description=None,
            output_units_description=None, frequency_range_start=None,
            frequency_range_end=None, frequency_range_db_variation=None):
        """

        response = station.channels[0].response
        instrument_sensitivity = response.instrument_sensitivity
        instrument_sensitivity.value = 1.
        instrument_sensitivity.frequency = 100.

        stages = response.response_stages
        # Insert OT geophone or accelerometer response in first stage of response:
        stages[0] = convert_pz_to_obspy(pzs)
        # Use generic digitizer for stage 2 with output sample rate = 6KHz
        stages[2].name = "Generic Digitizer = Placeholder for IMS Digitizer"
        stages[2].stage_gain = 1
        stages[2].decimation_input_sample_rate = Frequency(12000.)
        stages[2].decimation_factor = 2

        response.response_stages = stages[0:3]

        for channel in station.channels:
            channel.response = response

    inventory.write(xml_outfile, format='STATIONXML', nsmap={ns_tag: ns})

    return 1


def convert_pz_to_obspy(pz: instResp.polezero.polezero) -> PolesZerosResponseStage:
    ''' Convert internal polezero object to obspy PolesZeroResponseStage
    '''
    stage_sequence_number = 1
    stage_gain = pz.sensitivity
    stage_gain_frequency = pz.sensitivity_f
    normalization_factor = pz.a0
    normalization_frequency = pz.sensitivity_f

    zeros = pz.zeros
    poles = pz.poles

    if zeros is None:
        logger.debug("Inside convert_pz: zeros = None")
        zeros = []

    if pz.type == 'A':
        pz_transfer_function_type = "LAPLACE (RADIANS/SECOND)"
    elif pz.type == 'B':
        pz_transfer_function_type = "LAPLACE (HERTZ)"
    else:
        pz_transfer_function_type = "DIGITAL (Z-TRANSFORM)"

    input_units = pz.unitsIn
    output_units = pz.unitsOut
    pz_stage = PolesZerosResponseStage(stage_sequence_number,
                                       stage_gain,
                                       stage_gain_frequency,
                                       input_units,
                                       output_units,
                                       pz_transfer_function_type,
                                       normalization_frequency,
                                       zeros,
                                       poles,
                                       normalization_factor=normalization_factor,
                                       name=pz.name,
                                       )

    return pz_stage


def test_read_xml(xmlfile):

    inventory = Inventory.load_from_xml('OT.xml')

    for station in inventory.networks[0].stations:
        print(station.code, station.loc, station.sensor_id, station.extra.damping)

        for channel in station.channels:
            # channel.plot(min_freq=1., output=output)
            print(channel.code, channel.dip, channel.azimuth)

    return


def main():

    if 'SPP_COMMON' not in os.environ:
        logger.error("Set your SPP envs!")
        exit(2)

    path = os.environ['SPP_COMMON']
    xls_file = os.path.join(path, 'inventory_snapshot.xlsx')

    inventory = load_inventory_from_excel(xls_file)
    success = fix_OT_responses(inventory)
    inventory.write('OT.xml', format='STATIONXML', nsmap={ns_tag: ns})
    exit()

    sensor_file = os.path.join(path, 'sensors.csv')
    sensor_types_file = os.path.join(path, 'sensor_types.csv')
    cables_file = os.path.join(path, 'cables.csv')

    success = write_OT_xml(sensor_file, sensor_types_file, cables_file, xml_outfile='OT.xml')
    assert success == 1
    exit()

    # test_read_xml('OT.xml')
    test_print_OT_xml_summary('OT.xml')
    exit()

    test_read_stationxml('resources/ANMO.xml', 'ANMO2.xml')
    test_read_stationxml('resources/OT.xml', 'OT2.xml')
    test_read_csv_write_stationxml(sensor_csv, 'OT_new.xml')
    test_print_OT_xml_summary('OT_new.xml')

    return


if __name__ == "__main__":
    main()
