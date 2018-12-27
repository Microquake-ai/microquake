import obspy.core.inventory
from obspy.core.inventory.util import _unified_content_strings, _textwrap
from obspy.core.inventory.inventory import read_inventory
from obspy.core.inventory import Inventory, Network, Site
from obspy.core import AttribDict
from obspy.core.utcdatetime import UTCDateTime
import numpy as np

import csv
import copy
import os

from obspy.clients.nrl import NRL
nrl = NRL('http://ds.iris.edu/NRL/')

ns_tag='mq'
ns='MICROQUAKE'

"""
    Thin wrapper classes around obspy Station/Channel to provide property accesss to stn.x, stn.y, stn.z etc
    plus helper functions to convert between obspy Station, this Station and CSV records

    To maintain compatibility with obspy Inventory classes & methods, all the OT info is
    stored in {extra} dicts (accessible directly via properties), while the expected
    lat,lon,elev on station/channel are set to 0. for io compatibility.
"""

def read_csv(csv_file: str) -> []:
    stations = []
    with open(csv_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            station = {}
            #if line_count == 0:
                #print(f'Column names are {", ".join(row)}')

            station['code'] = row['Code']
            station['x'] = float(row['Easting'])
            station['y'] = float(row['Northing'])
            station['z'] = float(row['Elev'])
            station['loc'] = np.array([station['x'],station['y'],station['z']])
            station['long_name'] = row['Long Name']

            chans = [row[chan] for chan in ['Cmp01', 'Cmp02', 'Cmp03'] if row[chan].strip()]
            station['channels'] = []
            ic = 1
            for chan in chans:
                chan_dict = {}
                chan_dict['cmp'] = chan
                x = 'x%d' % ic
                y = 'y%d' % ic
                z = 'z%d' % ic
                chan_dict['orientation'] = np.array([float(row[x]), float(row[y]), float(row[z])])
                chan_dict['sensor_type'] = row['Type']
                station['channels'].append(chan_dict)
                ic += 1

            '''
            print("sta:%3s <%.3f, %.3f> %9.3f %s" % (row['Code'], float(row['Easting']), float(row['Northing']), \
                                                  float(row['Elev']), chans))
            '''
            line_count += 1
            stations.append(station)

    #print(f'Processed {line_count} lines.')
    return stations


class Station(obspy.core.inventory.station.Station):
    @classmethod
    def from_obspy_station(cls, obspy_station) -> obspy.core.inventory.Station :
        stn = obspy_station

        #     cls(*params) is same as calling Station(*params):
        sta = cls(stn.code, stn.latitude, stn.longitude, stn.elevation, channels=stn.channels, \
                  site=stn.site, vault=stn.vault, geology=stn.geology, equipments=stn.equipments, \
                  total_number_of_channels=stn.total_number_of_channels, \
                  selected_number_of_channels=stn.selected_number_of_channels, description=stn.description, \
                  restricted_status=stn.restricted_status, alternate_code=stn.alternate_code, \
                  comments=stn.comments, start_date=stn.start_date, end_date=stn.end_date, \
                  historical_code=stn.historical_code, data_availability=stn.data_availability)
        if getattr(stn, 'extra', None):
            sta.extra = copy.deepcopy(stn.extra)

        sta.channels = []
        for cha in stn.channels:
            sta.channels.append(Channel.from_obspy_channel(cha))

        return sta

    @classmethod
    def from_csv_station(cls, csv_station) -> obspy.core.inventory.Station :
        stn = csv_station

# New obspy seems to require creation_date .. here I set it before any expected event dats:
        sta = Station(stn['code'], 0., 0., 0., site=Site(name='Oyu Tolgoi'), creation_date=UTCDateTime("2015-12-31T12:23:34.5"))

        sta.extra = AttribDict({'x': { 'namespace': ns, 'value': stn['x'], },
                                'y': { 'namespace': ns, 'value': stn['y'], },
                                'z': { 'namespace': ns, 'value': stn['z'], },
                              })

        sta.channels = []
        for cha in stn['channels']:
            channel = Channel(code=cha['cmp'],    # required
                              location_code="",   # required
                              latitude = 0.,      # required
                              longitude= 0.,      # required
                              elevation= 0.,      # required
                              depth=0.,           # required
                              )

            channel.extra = AttribDict({ 'cos1': { 'namespace': ns, 'value': cha['orientation'][0] },
                                         'cos2': { 'namespace': ns, 'value': cha['orientation'][1] },
                                         'cos3': { 'namespace': ns, 'value': cha['orientation'][2] },
                                         'x':    { 'namespace': ns, 'value': stn['x'] },
                                         'y':    { 'namespace': ns, 'value': stn['y'] },
                                         'z':    { 'namespace': ns, 'value': stn['z'] },
                                         'sensor_type':    { 'namespace': ns, 'value': cha['sensor_type'].upper()},
                                      })
            # MTH: There doesn't seem to be any simple way to get the sensor_type (ACCELEROMETER vs GEOPHONE)
            #      to attach to the trace
            # e.g., tr.attach_response - just attaches a response object
            # The closest thing I can find is to set
            #   response.instrument.sensitivity.input_units to either "M/S**2" or "M/S"
            # Then will have to write a microquake method to detect these and/or use them for lookups

            # Temp attach a generic response to all OT channels:
            chan_dict = {}

            chan_dict['sensor'] = ['Sercel/Mark Products', 'L-22D', '325 Ohms', '1327 Ohms']
            chan_dict['datalogger'] = ['REF TEK','RT 130 & 130-SMA','1','100']
            response = nrl.get_response(sensor_keys=chan_dict['sensor'], datalogger_keys=chan_dict['datalogger'])

            if cha['sensor_type'].upper() == "ACCELEROMETER":
                input_units = "M/S**2"
            elif cha['sensor_type'].upper() == "GEOPHONE":
                input_units = "M/S" # The L-22D already has sensitivity.input_units = "M/S" ...

            response.instrument_sensitivity.input_units = input_units

            channel.response = response

            sta.channels.append(channel)

        sta.total_number_of_channels=len(sta.channels)
        sta.selected_number_of_channels=len(sta.channels)


        return sta

    @property
    def x(self):
        if self.extra:
            if self.extra.get('x', None):
                return float(self.extra.x.value)  # obspy inv_read converts everything in extra to str
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def y(self):
        if self.extra:
            if self.extra.get('y', None):
               return float(self.extra.y.value)
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def z(self):
        if self.extra:
            if self.extra.get('z', None):
                return float(self.extra.z.value)
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def loc(self):
        if self.extra:
            if self.extra.get('x', None) and self.extra.get('y', None) and self.extra.get('z', None):
                return np.array([self.x, self.y, self.z])
            else:
                raise AttributeError
        else:
            raise AttributeError

    '''
    @property
    def loc(self):
        return self.loc
        #return self._loc
    '''


    def __str__(self):
        contents = self.get_contents()

        x = self.latitude
        y = self.longitude
        z = self.elevation
        ret = ("Station {station_name}\n"
               "\tStation Code: {station_code}\n"
               "\tChannel Count: {selected}/{total} (Selected/Total)\n"
               "\t{start_date} - {end_date}\n"
               "\tAccess: {restricted} {alternate_code}{historical_code}\n"
               "\tLatitude: {x:.2f}, Longitude: {y:.2f}, "
               "Elevation: {z:.1f} m\n")

        if getattr(self, 'extra', None):
            if getattr(self.extra, 'x', None) and getattr(self.extra, 'y', None):
                x = self.x
                y = self.y
                z = self.z
                ret = ("Station {station_name}\n"
                    "\tStation Code: {station_code}\n"
                    "\tChannel Count: {selected}/{total} (Selected/Total)\n"
                    "\t{start_date} - {end_date}\n"
                    "\tAccess: {restricted} {alternate_code}{historical_code}\n"
                    "\tNorthing: {x:.2f}, Easting: {y:.2f}, "
                    "Elevation: {z:.1f} m\n")

        ret = ret.format(
            station_name=contents["stations"][0],
            station_code=self.code,
            selected=self.selected_number_of_channels,
            total=self.total_number_of_channels,
            start_date=str(self.start_date),
            end_date=str(self.end_date) if self.end_date else "",
            restricted=self.restricted_status,
            alternate_code="Alternate Code: %s " % self.alternate_code if self.alternate_code else "",
            historical_code="Historical Code: %s " % self.historical_code if self.historical_code else "",
            x=x, y=y, z=z)
        ret += "\tAvailable Channels:\n"
        ret += "\n".join(_textwrap(
            ", ".join(_unified_content_strings(contents["channels"])),
            initial_indent="\t\t", subsequent_indent="\t\t",
            expand_tabs=False))
        return ret

class Channel(obspy.core.inventory.channel.Channel):

    #__doc__ = obsevent.Origin.__doc__.replace('obspy', 'microquake')

    @classmethod
    def from_obspy_channel(cls, obspy_channel):
        chn = obspy_channel
        cha = Channel(chn.code, chn.location_code, chn.latitude, chn.longitude, chn.elevation, chn.depth, \
                      chn.azimuth, chn.dip, chn.types, chn.external_references, chn.sample_rate, \
                      chn.sample_rate_ratio_number_samples, chn.sample_rate_ratio_number_seconds, \
                      chn.storage_format, chn.clock_drift_in_seconds_per_sample, chn.calibration_units, \
                      chn.calibration_units_description, chn.sensor, chn.pre_amplifier, chn.data_logger, \
                      chn.equipment, chn.response, chn.description, chn.comments, chn.start_date, \
                      chn.end_date, chn.restricted_status, chn.alternate_code, chn.historical_code, \
                      chn.data_availability)

        if getattr(chn, 'extra', None):
            cha.extra = copy.deepcopy(chn.extra)


        return cha

    @property
    def sensor_type(self):
        if self.extra:
            if self.extra.get('sensor_type', None):
                return self.extra.sensor_type.value
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def cos1(self):
        if self.extra:
            if self.extra.get('cos1', None):
                return float(self.extra.cos1.value)
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def cos2(self):
        if self.extra:
            if self.extra.get('cos2', None):
                return float(self.extra.cos2.value)
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def cos3(self):
        if self.extra:
            if self.extra.get('cos3', None):
                return float(self.extra.cos3.value)
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def x(self):
        if self.extra:
            if self.extra.get('x', None):
               return float(self.extra.x.value)
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def y(self):
        if self.extra:
            if self.extra.get('y', None):
               return float(self.extra.y.value)
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def z(self):
        if self.extra:
            if self.extra.get('z', None):
               return float(self.extra.z.value)
            else:
                raise AttributeError
        else:
            raise AttributeError



def inv_station_list_to_dict(station_list):
    """ Convert station list (= list of microquake Station class metadata) to dict

        :param station_list: list of
        :return: dict
        :rtype: dict
    """
    sta_meta_dict = {}

    for station in station_list:
        dd={}
        dd['station'] = station.copy()
        dd['loc'] = station.loc

        chans_dict = {}
        for channel in station.channels:
            chans_dict[channel.code] = channel
        dd['chans'] = chans_dict
        dd['nchans'] = len(station.channels)

        sta_meta_dict[station.code] = dd

    return sta_meta_dict



def test_read_stationxml(xmlfile_in: str, xmlfile_out: str):
    """
        Read stationXML with or without namespace extras, wrap obspy Station/Channel in this class,
        then write back out.
    """
    inv = read_inventory(xmlfile_in)
    OT_stns = []
    for station in inv[0].stations:
        OT_stns.append(Station.from_obspy_station(station))
    inv[0].stations = OT_stns
    inv.write(xmlfile_out, format='STATIONXML', nsmap={ns_tag: ns})

def test_read_csv_write_stationxml(sensor_csv: str, xmlfile_out: str):
    """
        Read sensor.csv file, convert to wrapped obspy stationXML and write out
        Then read stationXML back in for good measure
    """
    stations = read_csv(sensor_csv)

    source = 'mth-test'         # Network ID of the institution sending the message.
    network = Network("OT")
    network.stations = []
    for station in stations:
        network.stations.append(Station.from_csv_station(station))

    inv = Inventory([network], source)
    inv.write(xmlfile_out, format='STATIONXML', nsmap={ns_tag: ns})

# TODO: Add test to be sure xml read in == xml written out
    inv_read = read_inventory(xmlfile_out)
    #inv_read = read_inventory('OT_bad.xml')

    OT_stns = []
    for station in inv_read[0].stations:
        OT_stns.append(Station.from_obspy_station(station))
    inv[0].stations = OT_stns
    #inv.write(xmlfile_out, format='STATIONXML', nsmap={ns_tag: ns})

def test_print_OT_xml_summary(xmlfile_in: str):
    """
        Simple test to read in an OT stationXML file (with extras confined to namespace)
            and manipulate/print them as local Station/Channel objects
    """
    inv = read_inventory(xmlfile_in)
    OT_stns = []
    for station in inv[0].stations:
        OT_stns.append(Station.from_obspy_station(station))

    for stn in OT_stns:
        print("%3s: <%.2f %.2f %8.2f>" % (stn.code, stn.y, stn.x, stn.z))
        for chan in stn.channels:
            print("  chan:%3s: <%.3f %.3f %.3f>" % (chan.code, chan.cos1, chan.cos2, chan.cos3))


def get_inventory(csv_file):
    stations = read_csv(csv_file)
    obspy_stations = []
    for station in stations:
        obspy_stations.append(Station.from_csv_station(station))

    source = 'mth-test'         # Network ID of the institution sending the message.
    network = Network("OT")
    network.stations = obspy_stations
    return Inventory([network], source)


def get_sensor_type_from_trace(tr):

    fname = 'get_sensor_type_from_trace'

    unit_map = {"DISP": ["M"],
                "VEL": ["M/S", "M/SEC"],
                "ACC": ["M/S**2", "M/(S**2)", "M/SEC**2", "M/(SEC**2)", "M/S/S"]
               }

    if 'response' in tr.stats:
        unit = None
        try:
            sensitivity = tr.stats.response.instrument_sensitivity
            i_u = sensitivity.input_units
        #except Exception:
        except AttributeError:
            print("%s: Couldn't find response.instrument_sensitivity.input_units" % fname)
            raise

        for key, value in unit_map.items():
            if i_u and i_u.upper() in value:
                unit = key
        if not unit:
            msg = ("ObsPy does not know how to map unit '%s' to "
                   "displacement, velocity, or acceleration - overall "
                   "sensitivity will not be recalculated.") % i_u
            raise ValueError(msg)

        return unit
    else:
        return 'not set'

#obs_Station.x = property(lambda self: self.latitude)
def main():

    if 'SPP_COMMON' not in os.environ:
        print("Set your SPP envs!")
        exit(2)

    sensor_csv = os.environ['SPP_COMMON'] + '/sensors.csv'
    test_read_csv_write_stationxml(sensor_csv, 'OT_new.xml')
    exit()

    inv = get_inventory(sensor_csv)
    for sta in inv[0].stations:
        print(sta)
    exit()

    test_read_stationxml('resources/ANMO.xml', 'ANMO2.xml')
    test_read_stationxml('resources/OT.xml', 'OT2.xml')
    test_read_csv_write_stationxml(sensor_csv, 'OT_new.xml')
    test_print_OT_xml_summary('OT_new.xml')


    exit()



if __name__ == "__main__":
    main()
