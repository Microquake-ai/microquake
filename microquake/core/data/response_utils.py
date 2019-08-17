import csv
import os
import pickle


def read_sensor_types_file(csv_file: str) -> []:
    """
    Read in a csv file

    :param csv_file: path to file
    :type: csv_file: str
    :return: sensor_types
    :rtype: dict
    """

    sensor_types = {}
    with open(csv_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:

            d = {}

            for k, v in row.items():
                #print("k=%s --> v=%s" % (k,v))
                d[k] = v

            sensor_types[row['sensor id']] = d

    # for sensor_id in sensor_types.keys():
        #print("sensor_id:%s --> %s" % (sensor_id, sensor_types[sensor_id]))

    return sensor_types


def read_cable_file(csv_file: str) -> []:
    """
    Read in a csv file

    :param csv_file: path to file
    :type: csv_file: str
    :return: sensor_types
    :rtype: dict
    """

    cables = {}
    with open(csv_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            v = list(row.values())
            cables[v[0]] = v[1]

    return cables


def write_NRL_dump_to_file(filename='resources/L-22D.response'):
    '''
    Note: This is the *only* function in this module that needs a
          network connection (to reach ds.iris.edu).  It only needs
          to be run when creating a *new* response template.
          Otherwise, the following function can be used to read in
          the response template from disk.
    '''
    from obspy.clients.nrl import NRL
    nrl = NRL('http://ds.iris.edu/NRL/')

    chan_dict = {}
    chan_dict['sensor'] = ['Sercel/Mark Products', 'L-22D', '325 Ohms', '1327 Ohms']
    chan_dict['datalogger'] = ['REF TEK', 'RT 130 & 130-SMA', '1', '100']
    response = nrl.get_response(sensor_keys=chan_dict['sensor'], datalogger_keys=chan_dict['datalogger'])

    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(response, output, pickle.HIGHEST_PROTOCOL)

    return


def read_NRL_from_dump(filename='resources/L-22D.response'):

    path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(path, filename)

    with open(filename, 'rb') as f:
        response = pickle.load(f)

    return response


def main():
    # write_NRL_dump_to_file()
    response = read_NRL_from_dump()

    for stage in response.response_stages:
        print(stage)

    return


if __name__ == '__main__':
    main()
