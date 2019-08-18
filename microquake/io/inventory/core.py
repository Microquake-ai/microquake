def read_csv(inventory_file: str) -> []:
    """
    Read in a station csv file, with companion sensor_file, sensor response
    file, and cable_file containing the cable response.

    :param csv_file: path to file
    :type: csv_file: str
    :param sensor_file: path to sensor response file
    :type: sensor_file: str
    :param cable_file: path to the cable response file
    :type: cable_file: str
    :return: stations
    :rtype: list
    """

    import numpy as np
    import pandas as pd
    import obspy.core.inventory
    from obspy.core.util.attribdict import AttribDict
    from obspy.signal.invsim import paz_to_freq_resp, corn_freq_2_paz
    from obspy.core.inventory import (InstrumentPolynomial, Response,
                                      ResponseStage, InstrumentSensitivity,
                                      PolesZerosResponseStage)
    ns = 'MICROQUAKE'

    with open(inventory_file, mode='r') as csv_file:
        sensor_file = csv_file.readline().split(',')[1]
        cable_file = csv_file.readline().split(',')[1]

    sensors = pd.read_csv(sensor_file)
    cables = pd.read_csv(cable_file)
    inventory = pd.read_csv(inventory_file, header=2)

    for i, station in inventory.iterrows():
        location_code = station['Location Code']
        obs_station = obspy.core.inventory.Station(station['Code'], 0, 0, 0)

        obs_station.extra = AttribDict({'x': { 'namespace': ns,
                                              'value': station['x']},
                                        'y': { 'namespace': ns,
                                              'value': station['y']},
                                        'z': { 'namespace': ns,
                                               'value': station['z']}})

        ch_code = ""
        if sensors[station['type']]['sensor type'].lower() == 'geophone':
            ch_code = 'PE'
        elif sensors[station['type']]['sensor type'].lower() == \
                'accelerometer':
            ch_code = 'AE'
        channels= []
        for j in range(0, 3):
            if station['cmp0%d' % j]:
                ch_code += station['cmp0%d' % j].upper()
                ori_x = station['y%d' % j]
                ori_y = station['y%d' % j]
                ori_z = station['y%d' % j]
                ori_h = np.linalg.norm(ori_y - ori_x)
                azimuth = np.arctan2(ori_x, ori_y) / np.pi * 180
                dip = np.arctan2(ori_z, ori_h) / np.pi * 180

                if station['type'].lower() == 'geophone':
                    sensor = sensors[sensors['sensor id'] ==
                                     station['Sensor Type ID']]
                    paz = corn_freq_2_paz(sensor['resonance frequency (Hz)'],
                                          damp=sensor['damping'],
                                          gain=sensor['gain'],
                                          sensitivity=sensor['sensitivity'])
                    i_s = InstrumentSensitivity(1, 14, input_units='M/S',
                                                output_units='M/S',
                                                input_units_description='velocity',
                                                output_units_description='velocity')


                    if station['Cable Type']:
                        C = cables[cables['cable id'] ==
                                   station['Cable Type']]['capacity (pF/m)']
                        l = station['Cable Length']
                        R = sensor['output resistance (ohm)']
                        paz.gain *= 1 / (R * l * C)
                        paz['poles'].append(- 1 / (R * l *C))

                    pzrs = PolesZerosResponseStage(1, 1, 14, 'M/S', 'M/S',
                                                   'LAPLACE (RADIANT/SECOND)',
                                                   1, paz['zeros'],
                                                   paz['poles'])

                    res = Response(instrument_sensitivity=i_s,
                                   response_stages=[pzrs])





                elif station['type'].lower() == 'accelerometer':
                    sensor = sensors[sensors['sensor id'] ==
                                     station['Sensor Type ID']]
                    i_s = InstrumentSensitivity(1, 14, input_units='M/S/S',
                                                output_units='M/S',
                                                input_units_description='velocity',
                                                output_units_description='velocity')

                    paz = corn_freq_2_paz(sensor['resonance frequency (Hz)'],
                                          damp=sensor['damping'],
                                          gain=sensor['gain'],
                                          sensitivity=sensor['sensitivity'])

                    pzrs = PolesZerosResponseStage(1, 1, 14, 'M/S/S', 'M/S',
                                                   'LAPLACE (RADIANT/SECOND)',
                                                   1, paz['zeros'],
                                                   paz['poles'])


                    paz['zeros'] = [0]

                    res = Response(instrument_sensitivity=i_s,
                                   response_stages=[pzrs])

                obs_channel = obspy.core.inventory.Channel(ch_code,
                                                           location_code,
                                                            0, 0, 0, 0,
                                                           azimuth=azimuth,
                                                           dip=dip,
                                                           response=res)

                # JPM
                # 1 - convert to microquake.core.inventory.channel
                # 2 - append to channel object
                # 3 - add the channels to obspy.core.inventory.Station
                # 4 - convert to microquake.core.inventory.Station






    #     if station['cmp02']:
    #         ncomponent = 3
    #     else:
    #         ncomponent = 1
    #     obspy.core.inventory.Channel()
    #
    # sensors = []
    # with open (sensor_file, mode='r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     for row in csv_reader:
    #         sensor = {}
    #         sensor['id'] = row['sensor id']
    #         sensor['sensor type'] = row['sensor type']
    #         sensor['part number'] = row['part number']
    #         sensor['output resistance'] = float(row['output resistance (ohm)'])
    #         sensor['gain'] = float(row['gain'])
    #         sensor['sensitivity'] = float(row['sensitivity'])
    #         sensor['resonance frequency'] = float(row['resonance frequency ('
    #                                                   'Hz)'])
    #         sensor['damping'] = float(row['damping'])
    #         sensors.append(sensor)
    #
    # cables = []
    # with open (cable_file, mode='r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     for row in csv_reader:
    #         cable = {}
    #         cable['id'] = row['cable id']
    #         cable['capacity'] = float(row['capacity (pF/m)'])
    #         cables.append(cable)
    #
    #
    # stations = []
    # with open(inventory_file, mode='r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     line_count = 0
    #     for row in csv_reader:
    #         station = {}
    #         #if line_count == 0:
    #             #print(f'Column names are {", ".join(row)}')
    #
    #         station['code'] = row['Code']
    #         station['x'] = float(row['Easting'])
    #         station['y'] = float(row['Northing'])
    #         station['z'] = float(row['Elev'])
    #         station['loc'] = np.array([station['x'],station['y'],station['z']])
    #         station['long_name'] = row['Long Name']
    #
    #         chans = [row[chan] for chan in ['Cmp01', 'Cmp02', 'Cmp03']
    #                  if row[chan].strip()]
    #         station['channels'] = []
    #         ic = 1
    #         for chan in chans:
    #             chan_dict = {}
    #             chan_dict['cmp'] = chan
    #             x = 'x%d' % ic
    #             y = 'y%d' % ic
    #             z = 'z%d' % ic
    #             chan_dict['orientation'] = np.array([float(row[x]),
    #                                                  float(row[y]),
    #                                                  float(row[z])])
    #             chan_dict['sensor_type'] = row['Type']
    #             station['channels'].append(chan_dict)
    #             ic += 1
    #
    #
    #
    #         line_count += 1
    #         stations.append(station)
    #
    # return stations
