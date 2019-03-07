
import csv


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

            for k,v in row.items():
                #print("k=%s --> v=%s" % (k,v))
                d[k]=v

            sensor_types[row['sensor id']] = d

    #for sensor_id in sensor_types.keys():
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
