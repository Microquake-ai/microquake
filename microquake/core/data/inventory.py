import copy
import csv
import os

import numpy as np
import obspy.core.inventory
from obspy.core import AttribDict
from obspy.core.inventory import Network
from obspy.core.inventory.inventory import read_inventory
from obspy.core.inventory.util import (Equipment, Operator, Person, PhoneNumber, Site, _textwrap,
                                       _unified_content_strings)
from obspy.core.utcdatetime import UTCDateTime

from microquake.core.data.response_utils import read_NRL_from_dump

ns_tag = 'mq'
ns = 'MICROQUAKE'

"""
    Thin wrapper classes around obspy Station/Channel to provide property accesss to stn.x, stn.y, stn.z etc
    plus helper functions to convert between obspy Station, this Station and CSV records

    To maintain compatibility with obspy Inventory classes & methods, all the OT info is
    stored in {extra} dicts (accessible directly via properties), while the expected
    lat,lon,elev on station/channel are set to 0. for io compatibility.
"""


def load_inventory_from_excel(xls_file: str) -> 'microquake.core.data.inventory.Inventory':
    '''
    Read in a multi-sheet excel file with network metadata sheets:
        Sites, Networks, Hubs, Stations, Components, Sensors, Cables, Boreholes
    Organize these into a microquake Inventory object

    Note: The returned channel responses will be *generic* (= L-22D).
          To replace these with actual (calculated) responses, pass the inventory on to:
          write_ot.fix_OT_responses(inventory)
          (requires libInst module be installed!)

    :param xls_file: path to excel file
    :type: xls_file: str
    :return: inventory
    :rtype: microquake.core.data.inventory.Inventory
    '''

    fname = 'load_inventory_from_excel'

# This function requires pandas:
    import importlib.util
    import sys

    spec = importlib.util.find_spec('pandas')

    if spec is None:
        print("%s: pandas is not installed --> Install and try again" % fname)

        return

    import pandas as pd

    # read_excel Returns an OrderedDict
    df_dict = pd.read_excel(xls_file, sheet_name=None)

    # Initialize Inventory object containing Network from Sites+Networks sheets:
    '''
Sites:
   code coordinate_system country    description   id       name   operator timezone
   OT   Cartesian/Local  Mongolia  Added Manually   1  Oyu Tolgoi  Rio Tinto   +08:00

Networks:
   code                contact_email           contact_name   ...    id                            name  site_id
   HNUG  jean-philippem@riotinto.com  Jean-Philippe Mercier   ...     1  Hugo North Underground Network        1

    '''
# source (str) Network ID of the institution sending the message.
    source = df_dict['Sites'].iloc[0]['code']
# sender (str, optional) Name of the institution sending this message.
    sender = df_dict['Sites'].iloc[0]['operator']
    net_code = df_dict['Networks'].iloc[0]['code']
    net_descriptions = df_dict['Networks'].iloc[0]['name']

    contact_name = df_dict['Networks'].iloc[0]['contact_name']
    contact_email = df_dict['Networks'].iloc[0]['contact_email']
    contact_phone = df_dict['Networks'].iloc[0]['contact_phone']
    site_operator = df_dict['Sites'].iloc[0]['operator']
    site_country = df_dict['Sites'].iloc[0]['country']
    site_name = df_dict['Sites'].iloc[0]['name']
    site_code = df_dict['Sites'].iloc[0]['code']

    print("source=%s" % source)
    print("sender=%s" % sender)
    print("net_code=%s" % net_code)

    network = Network(net_code)
    inventory = Inventory([network], source)

# MTH: obspy requirements for PhoneNumber are super specific:
# So likely this will raise an error if/when someone changes the value in Networks.contact_phone
    '''
    PhoneNumber(self, area_code, phone_number, country_code=None, description=None):
        :type area_code: int
        :param area_code: The area code.
        :type phone_number: str
        :param phone_number: The phone number minus the country and area code.
            Must be in the form "[0-9]+-[0-9]+", e.g. 1234-5678.
        :type country_code: int, optional
        :param country_code: The country code.
    '''

    import re
    phone = re.findall(r"[\d']+", contact_phone)
    area_code = int(phone[0])
    number = "%s-%s" % (phone[1], phone[2])
    phonenumber = PhoneNumber(area_code=area_code, phone_number=number)

    person = Person(names=[contact_name], agencies=[site_operator], emails=[contact_email], phones=[phonenumber])
    operator = Operator(agencies=[site_operator], contacts=[person])
    site = Site(name=site_name, description=site_name, country=site_country)

    # Merge Stations+Components+Sensors+Cables info into sorted stations + channels dicts:

    df = df_dict['Stations']
    df_merge = pd.merge(df_dict['Stations'], df_dict['Components'],
                        left_on='id', right_on='station_id',
                        how='inner', suffixes=('', '_channel'))

    df_merge2 = pd.merge(df_merge, df_dict['Sensors'],
                         left_on='sensor_id', right_on='id',
                         how='inner', suffixes=('', '_sensor'))

    df_merge3 = pd.merge(df_merge2, df_dict['Cables'],
                         left_on='cable_id', right_on='id',
                         how='inner', suffixes=('', '_cable'))

    df = df_merge3.sort_values('id')

    # Need to sort by unique station codes, then look through 1-3 channels to add
    stn_codes = set(df['code'])
    stations = []

    for code in stn_codes:
        chan_rows = df.loc[df['code'] == code]
        row = chan_rows.iloc[0]

        station = {}
    # Set some keys explicitly
        station['code'] = row['code']
        station['x'] = row['location_x']
        station['y'] = row['location_y']
        station['z'] = row['location_z']
        station['loc'] = np.array([station['x'], station['y'], station['z']])
        station['long_name'] = row['name']
    # MTH: 2019/07 Seem to have moved from pF to F on Cables sheet:
        station['cable_capacitance_pF_per_meter'] = row['c'] * 1e12

    # Set the rest (minus empty fields) directly from spreadsheet names:
        renamed_keys = {'code', 'location_x', 'location_y', 'location_z', 'name'}
        # These keys are either redundant or specific to channel, not station:
        remove_keys = {'code_channel', 'id_channel', 'orientation_x', 'orientation_y', 'orientation_z',
                       'id_sensor', 'enabled_channel', 'station_id', 'id_cable'}
        keys = row.keys()
        empty_keys = keys[pd.isna(row)]
        keys = set(keys) - set(empty_keys) - renamed_keys - remove_keys

        for key in keys:
            station[key] = row[key]

    # Added keys:
        station['motion'] = 'VELOCITY'

        if row['sensor_type'].upper() == 'ACCELEROMETER':
            station['motion'] = 'ACCELERATION'

    # Attach channels:
        station['channels'] = []

        for index, r in chan_rows.iterrows():
            chan = {}
            chan['cmp'] = r['code_channel'].lower()
            chan['orientation'] = np.array([r['orientation_x'], r['orientation_y'], r['orientation_z']])
            chan['enabled'] = r['enabled_channel']
            station['channels'].append(chan)
        stations.append(station)

    # Convert these station dicts to inventory.Station objects and attach to inventory.network:
    obspy_stations = []

    for station in stations:
        # This is where namespace is first employed:
        obspy_station = Station.from_csv_station(station)
        obspy_station.site = site
        obspy_station.operators = [operator]
        obspy_stations.append(obspy_station)
        # obspy_stations.append(Station.from_csv_station(station))
        # print(obspy_station.operators[0].contacts[0].names)
        # print(obspy_station.site.country)
        #print(obspy_station.code, obspy_station.equipments[0].manufacturer, obspy_station.equipments[0].model)

    network.stations = obspy_stations

    return inventory


def read_csv(csv_file: str) -> []:
    """
    Read in a station csv file

    Note: This should probably only be called by write_OT once to create stationXML from csv files.
          After that, use:
          >>>inventory = Inventory.load_from_xml('OT.xml')

    :param csv_file: path to file
    :type: csv_file: str
    :return: stations
    :rtype: list
    """

    stations = []
    with open(csv_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0

        for row in csv_reader:
            station = {}
            # if line_count == 0:
            #print(f'Column names are {", ".join(row)}')

            station['code'] = row['Code']
            station['x'] = float(row['Easting'])
            station['y'] = float(row['Northing'])
            station['z'] = float(row['Elev'])
            station['loc'] = np.array([station['x'], station['y'], station['z']])
            station['long_name'] = row['Long Name']
            station['sensor_id'] = row['Type']
            #station['sensor_type'] = row['Type']
            station['cable_type'] = row['Cable Type']
            station['cable_length'] = row['Cable length']
            station['motion'] = row['Motion']

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
                station['channels'].append(chan_dict)
                ic += 1
            print(chan_dict)
            exit()

            '''
            print("sta:%3s <%.3f, %.3f> %9.3f %s" % (row['Code'], float(row['Easting']), float(row['Northing']), \
                                                  float(row['Elev']), chans))
            '''
            line_count += 1
            stations.append(station)

    #print(f'Processed {line_count} lines.')

    return stations


class Inventory(obspy.core.inventory.inventory.Inventory):

    @classmethod
    def load_from_xml(cls, stationXMLfile):
        '''
            Load stationXMLfile and return a microquake Inventory object
        '''

        source = 'mth-test'         # Network ID of the institution sending the message.

        obspy_inv = read_inventory(stationXMLfile)

        for network in obspy_inv.networks:
            stations = []

            for station in network.stations:
                stations.append(Station.from_obspy_station(station))
            network.stations = stations

        return Inventory([network], source)

    def get_station(self, sta):
        return self.select(sta)

    def get_channel(self, sta=None, cha=None):
        return self.select(sta, cha_code=cha)

    def select(self, sta_code, net_code=None, cha_code=None):
        '''
            Select a single Station or Channel object out of the Inventory
        '''
        station_found = None

        for network in self:
            for station in network.stations:
                if station.code == sta_code:
                    if net_code:
                        if network.code == net_code:
                            station_found = station

                            break
                    else:
                        station_found = station

                        break

        if not station_found:
            return None

        channel_found = None

        if cha_code:
            for cha in station_found.channels:
                if cha.code == cha_code:
                    channel_found = cha

                    break

            return channel_found

        else:
            return station_found

    '''
        MTH: Note that stations is an attribute on obspy.inventory.Network and returns a list:
            >for station in inventory.network[0].stations:

        Here we add a convenience function that can be used as:
            >for station in inventory.stations():
    '''

    def stations(self, net_code=None):
        '''
            Return list of all Station objects in Inventory, optionally
                with matching net_code
        '''

        stations = []

        for network in self:
            if net_code:
                if network.code == net_code:
                    for station in network.stations:
                        stations.append(station)
            else:
                for station in network.stations:
                    stations.append(station)

        return stations

    def get_sta_codes(self, unique=False):

        sta_codes = []

        for network in self:
            for station in network.stations:
                sta_codes.append(station.code)

        if unique:
            return set(sta_codes)
        else:
            return sta_codes

    def sort_by_motion(self, motion='VELOCITY'):

        stations = []

        for network in self:
            for station in network.stations:
                if station.motion.upper() == motion.upper():
                    stations.append(station)

        return stations

    def get_sensor_types(self):

        sensors = []

        for network in self:
            for station in network.stations:
                sensors.append(station.sensor_type)

        return set(sensors)


class Station(obspy.core.inventory.station.Station):
    @classmethod
    def from_obspy_station(cls, obspy_station) -> obspy.core.inventory.Station:
        stn = obspy_station

        #     cls(*params) is same as calling Station(*params):
        sta = cls(stn.code, stn.latitude, stn.longitude, stn.elevation, channels=stn.channels,
                  site=stn.site, vault=stn.vault, geology=stn.geology, equipments=stn.equipments,
                  total_number_of_channels=stn.total_number_of_channels,
                  selected_number_of_channels=stn.selected_number_of_channels, description=stn.description,
                  restricted_status=stn.restricted_status, alternate_code=stn.alternate_code,
                  comments=stn.comments, start_date=stn.start_date, end_date=stn.end_date,
                  historical_code=stn.historical_code, data_availability=stn.data_availability)

        if getattr(stn, 'extra', None):
            sta.extra = copy.deepcopy(stn.extra)

        sta.channels = []

        for cha in stn.channels:
            sta.channels.append(Channel.from_obspy_channel(cha))

        return sta

    @classmethod
    def from_csv_station(cls, csv_station) -> obspy.core.inventory.Station:
        stn = csv_station

# New obspy seems to require creation_date .. here I set it before any expected event dats:
        #sta = Station(stn['code'], 0., 0., 0., site=Site(name='Oyu Tolgoi'), creation_date=UTCDateTime("2015-12-31T12:23:34.5"))
# Putting the OT station long_name into obspy Station historical_code:
        '''
        class Equipment(type=None, description=None, manufacturer=None, vendor=None, model=None, serial_number=None,.
                    installation_date=None, removal_date=None, calibration_dates=None, resource_id=None)
        '''
        equipment = None

        if 'manufacturer' in stn:
            equipment = Equipment(type='Sensor', manufacturer=stn['manufacturer'], model=stn['model'])

        sta = Station(stn['code'], 0., 0., 0., site=Site(name='Oyu Tolgoi'),
                      equipments=[equipment],
                      historical_code=stn['long_name'],
                      creation_date=UTCDateTime("2015-12-31T12:23:34.5"),
                      start_date=UTCDateTime("2015-12-31T12:23:34.5"),
                      end_date=UTCDateTime("2599-12-31T12:23:34.5"))

        non_extras_keys = {'code', 'long_name', 'channels', 'start_date', 'end_date'}
        keys = stn.keys() - non_extras_keys
        sta.extra = AttribDict()

        for key in keys:
            sta.extra[key] = {'namespace': ns, 'value': stn[key]}
        '''
        sta.extra = AttribDict({'x': { 'namespace': ns, 'value': stn['x'], },
                                'y': { 'namespace': ns, 'value': stn['y'], },
                                'z': { 'namespace': ns, 'value': stn['z'], },
                                'sensor_id':    { 'namespace': ns, 'value': stn['sensor_id']},
                                #'sensor_type':    { 'namespace': ns, 'value': stn['sensor_type'].upper()},
                                'motion':         { 'namespace': ns, 'value': stn['motion'].upper()},
                                'cable_type':     { 'namespace': ns, 'value': stn['cable_type']},
                                'cable_length':   { 'namespace': ns, 'value': stn['cable_length'] },
                              })
        '''

        sta.channels = []

        for cha in stn['channels']:
            channel = Channel(code=cha['cmp'],    # required
                              location_code="",   # required
                              latitude=0.,      # required
                              longitude=0.,      # required
                              elevation=0.,      # required
                              depth=0.,           # required
                              start_date=UTCDateTime("2015-12-31T12:23:34.5"),
                              end_date=UTCDateTime("2599-12-31T12:23:34.5"),
                              )

            channel.extra = AttribDict({'cos1': {'namespace': ns, 'value': cha['orientation'][0]},
                                        'cos2': {'namespace': ns, 'value': cha['orientation'][1]},
                                        'cos3': {'namespace': ns, 'value': cha['orientation'][2]},
                                        'cosines': {'namespace': ns, 'value': cha['orientation']},
                                        'x':    {'namespace': ns, 'value': stn['x']},
                                        'y':    {'namespace': ns, 'value': stn['y']},
                                        'z':    {'namespace': ns, 'value': stn['z']},
                                        })

            cosines = channel.extra['cosines'].value
            #cosines = np.array([0, 0, 1])
            #cosines = np.array([1, 0, 1]) * 1./np.sqrt(2)

            channel.dip, channel.azimuth = get_dip_and_azimuth_from_cosines(cosines)

            # MTH: There doesn't seem to be any simple way to get the sensor_type (ACCELEROMETER vs GEOPHONE)
            #      to attach to the trace
            # e.g., tr.attach_response - just attaches a response object
            # The closest thing I can find is to set
            #   response.instrument.sensitivity.input_units to either "M/S**2" or "M/S"
            # Then will have to write a microquake method to detect these and/or use them for lookups

            # Temp attach a generic response to all OT channels to use as template:
            response = read_NRL_from_dump(filename='resources/L-22D.response')

            if stn['motion'].upper() == 'ACCELERATION':
                input_units = "M/S**2"
                input_units_description = "Acceleration in Meters per Second**2"
            elif stn['motion'].upper() == 'VELOCITY':
                input_units = "M/S"
                input_units_description = "Velocity in Meters per Second"
            else:
                print("Unknown motion=[%s]" % stn['motion'])
                exit()

            response.instrument_sensitivity.input_units = input_units
            response.instrument_sensitivity.input_units_description = input_units_description

            channel.response = response

            sta.channels.append(channel)

        sta.total_number_of_channels = len(sta.channels)
        sta.selected_number_of_channels = len(sta.channels)

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

    @property
    def sensor_id(self):
        if self.extra:
            if self.extra.get('sensor_id', None):
                return self.extra.sensor_id.value
            else:
                raise AttributeError
        else:
            raise AttributeError

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
    def motion(self):
        if self.extra:
            if self.extra.get('motion', None):
                return self.extra.motion.value
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def cable_type(self):
        if self.extra:
            if self.extra.get('cable_type', None):
                return self.extra.cable_type.value
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def cable_length(self):
        if self.extra:
            if self.extra.get('cable_length', None):
                return float(self.extra.cable_length.value)
            else:
                raise AttributeError
        else:
            raise AttributeError

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
        cha = Channel(chn.code, chn.location_code, chn.latitude, chn.longitude, chn.elevation, chn.depth,
                      chn.azimuth, chn.dip, chn.types, chn.external_references, chn.sample_rate,
                      chn.sample_rate_ratio_number_samples, chn.sample_rate_ratio_number_seconds,
                      chn.storage_format, chn.clock_drift_in_seconds_per_sample, chn.calibration_units,
                      chn.calibration_units_description, chn.sensor, chn.pre_amplifier, chn.data_logger,
                      chn.equipment, chn.response, chn.description, chn.comments, chn.start_date,
                      chn.end_date, chn.restricted_status, chn.alternate_code, chn.historical_code,
                      chn.data_availability)

        if getattr(chn, 'extra', None):
            cha.extra = copy.deepcopy(chn.extra)

        return cha

    """
    @property
    def azimuth(self):
        if self.extra:
            if self.extra.get('cosines', None):
                return np.arctan2(cosines[0], cosines[1]) * 180./np.pi
            else:
                raise AttributeError
        else:
            raise AttributeError

    @property
    def dip(self):
        if self.extra:
            if self.extra.get('cosines', None):
                hlen = np.linalg.norm(cosines[:2])
                return -np.arctan2(cosines[2], hlen) * 180./np.pi
            else:
                raise AttributeError
        else:
            raise AttributeError
    """

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
    def cosines(self):
        if self.extra:
            # if self.extra.get('cosines', None):

            if self.extra.get('cos1', None) and self.extra.get('cos2', None) and self.extra.get('cos3', None):
                return np.array([self.cos1, self.cos2, self.cos3])
                # return float(self.extra.cos.value)
                # return self.extra.cosines.value
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


# def inv_station_list_to_dict(station_list):
def inv_station_list_to_dict(inventory):
    """ Convert station list (= list of microquake Station class metadata) to dict

        :param station_list: list of
        :return: dict
        :rtype: dict
    """
    sta_meta_dict = {}

    station_list = []

    for i in range(len(inventory)):
        net = inventory[i]
        station_list += net.stations

    for station in station_list:
        dd = {}
        dd['station'] = station.copy()

        # if 'loc' in station:

        if hasattr(station, 'loc'):
            dd['loc'] = station.loc
        else:
            dd['lat'] = station.latitude
            dd['lon'] = station.longitude
            dd['elev'] = station.elevation

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


def load_inventory(fname, format='CSV', **kwargs):
    """
    Note: This is mostly deprecated.
          A better approach is to use write_OT.py to create stationXML (e.g. 'OT.xml')
          from the various csv files, and just read in this static xml file
          >>> inventory = Inventory.load_from_xml('OT.xml')

    An obspy inventory is just a list of stations contained in a list of networks
    This will return such a list, only the contained stations are
    microquake.core.data.station2.Station class

    :param fname: path to file
    :type: fname: str
    :param format: input file type (CSV only type for now)
    :type: format: str
    :return: station inventory = list of networks containing list of (this) Station class
    :rtype: obspy.core.inventory

    """

    if format == 'CSV':
        stations = read_csv(fname)
    else:
        print("load_inventory: Only set up to read CSV formats!!!")

        return None
    obspy_stations = []

    for station in stations:
        obspy_stations.append(Station.from_csv_station(station))

    source = 'mth-test'         # Network ID of the institution sending the message.
    network = Network("OT")
    network.stations = obspy_stations

    return Inventory([network], source)


def get_corner_freq_from_pole(pole):
    '''
        get distance [rad/s] from lowest order pole to origin
            and return Hz [/s]
    '''

    return np.sqrt(pole.real**2 + pole.imag**2) / (2.*np.pi)


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
        # except Exception:
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

    return


def get_dip_and_azimuth_from_cosines(cosines):
    """
    MTH: My understanding of the orientation is this:
         x=E, y=N, z=Up  -eg, this *is* a right-handed coord sys.
         Most seismology has x=N and z=Down, hence the confusion.
         In our convention, Z also works as "elevation", in that elevation increases positive upwards.

         The way this works with the channel direction cosines, is:
         [0, 0, 1] = a channel aligned with the +ve Z direction, will have a
         dip = -90 deg. This is the same as traditional convention (see below)

        From SEED manual, Here are traditional channel orientations:
        Z — Dip -90, Azimuth 0 (Reversed: Dip 90, Azimuth 0)
        N — Dip 0, Azimuth 0 (Reversed: Dip 0, Azimuth 180)
        E — Dip 0, Azimuth 90 (Reversed: Dip 0, Azimuth 270)

    Examples:
                            E  N  Z
        >>>cosines = np.array([0, 0, 1]) --> azimuth=0, dip=-90 (aligned with +ve=up vertical dir)
        >>>cosines = np.array([1, 0, 1]) * 1./np.sqrt(2) --> azimuth=90, dip=-45 deg
        >>>cosines = np.array([-1, 1, -np.sqrt(2)]) * 1./np.sqrt(4) --> azimuth=315, dip=45
    """

    azimuth = np.arctan2(cosines[0], cosines[1]) * 180./np.pi
    hlen = np.linalg.norm(cosines[:2])
    dip = -np.arctan2(cosines[2], hlen) * 180./np.pi

    # obspy/stationXML will enforce: azimuth:0-360 deg, dip:-90 to 90 deg.

    if azimuth < 0:
        azimuth += 360.

    assert dip >= -90 and dip <= 90.
    assert azimuth >= 0 and azimuth <= 360.

    return dip, azimuth


def get_cosines_from_dip_and_azimuth(_dip, _az):
    az = _az * np.pi/180.
    dip = _dip * np.pi/180.

    return np.array([np.sin(az) * np.cos(-dip), np.cos(az) * np.cos(-dip), np.sin(-dip)])


def main():

    inventory = load_inventory_from_excel('inventory_snapshot.xlsx')
    #success = fix_OT_responses(inv)

    #inventory = Inventory.load_from_xml('OT.xml')

    for station in inventory.networks[0].stations:
        print(station.code, station.loc, station.sensor_id, station.extra.damping)

    exit()

    #inventory = load_inventory(os.environ['SPP_COMMON'] + '/sensors.csv')
    #sensor_csv = os.environ['SPP_COMMON'] + '/sensors.csv'
    #test_read_csv_write_stationxml(sensor_csv, 'OT_new.xml')

    test_read_stationxml('resources/ANMO.xml', 'ANMO2.xml')
    test_read_stationxml('resources/OT.xml', 'OT2.xml')
    test_read_csv_write_stationxml(sensor_csv, 'OT_new.xml')
    test_print_OT_xml_summary('OT_new.xml')

    return


if __name__ == "__main__":
    main()
