import obspy.core.inventory
from obspy.core.inventory import Network, Site
from obspy.core.inventory.inventory import read_inventory
from obspy.core.inventory.util import _unified_content_strings, _textwrap
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
    """
    Read in a station csv file

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
            #if line_count == 0:
                #print(f'Column names are {", ".join(row)}')

            station['code'] = row['Code']
            station['x'] = float(row['Easting'])
            station['y'] = float(row['Northing'])
            station['z'] = float(row['Elev'])
            station['loc'] = np.array([station['x'],station['y'],station['z']])
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
                if station.motion.upper()  == motion.upper():
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
        #sta = Station(stn['code'], 0., 0., 0., site=Site(name='Oyu Tolgoi'), creation_date=UTCDateTime("2015-12-31T12:23:34.5"))
# Putting the OT station long_name into obspy Station historical_code:
        sta = Station(stn['code'], 0., 0., 0., site=Site(name='Oyu Tolgoi'), \
                      historical_code=stn['long_name'], \
                      creation_date=UTCDateTime("2015-12-31T12:23:34.5"))

        sta.extra = AttribDict({'x': { 'namespace': ns, 'value': stn['x'], },
                                'y': { 'namespace': ns, 'value': stn['y'], },
                                'z': { 'namespace': ns, 'value': stn['z'], },
                                'sensor_id':    { 'namespace': ns, 'value': stn['sensor_id']},
                                #'sensor_type':    { 'namespace': ns, 'value': stn['sensor_type'].upper()},
                                'motion':         { 'namespace': ns, 'value': stn['motion'].upper()},
                                'cable_type':     { 'namespace': ns, 'value': stn['cable_type']},
                                'cable_length':   { 'namespace': ns, 'value': stn['cable_length'] },
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
                                         'cosines': { 'namespace': ns, 'value': cha['orientation'] },
                                         'x':    { 'namespace': ns, 'value': stn['x'] },
                                         'y':    { 'namespace': ns, 'value': stn['y'] },
                                         'z':    { 'namespace': ns, 'value': stn['z'] },
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

            # Temp attach a generic response to all OT channels:
            chan_dict = {}

            chan_dict['sensor'] = ['Sercel/Mark Products', 'L-22D', '325 Ohms', '1327 Ohms']
            chan_dict['datalogger'] = ['REF TEK','RT 130 & 130-SMA','1','100']
            response = nrl.get_response(sensor_keys=chan_dict['sensor'], datalogger_keys=chan_dict['datalogger'])
            #print(response.get_sacpz())
            #print(response.get_paz())

            from obspy.core.inventory.response import ResponseStage

            '''
            stage1 = response.response_stages[0]
            stage2 = response.response_stages[1]
            stage3 = ResponseStage(3,1.0, .05, "COUNTS", "V")
            #stage3 = ResponseStage(3,1.0, .05, "COUNTS", "V", decimation_input_sample_rate=200)
                                   #decimation_factor=1, decimation_offset=0, decimation_delay=0,
                                   #decimation_correction=0)

            response.response_stages = [stage1, stage2]
            #response.response_stages = [stage1, stage2, stage3]

            for stage in response.response_stages:
                print(stage)
                print(stage.input_units, stage.output_units)
            response.plot(output="VEL", min_freq=.01, sampling_rate=100.)
            exit()
            '''

            if stn['motion'].upper() == 'ACCELERATION':
                input_units = "M/S**2"
                input_units_description = "Acceleration in Meters per Second**2"
            elif stn['motion'].upper() == 'VELOCITY':
                input_units = "M/S"
                input_units_description = "Velocity in Meters per Second"

            response.instrument_sensitivity.input_units = input_units
            response.instrument_sensitivity.input_units_description = input_units_description

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
            if self.extra.get('cosines', None):
                #return float(self.extra.cos.value)
                return self.extra.cosines.value
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




#def inv_station_list_to_dict(station_list):
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
        dd={}
        dd['station'] = station.copy()

        #if 'loc' in station:
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
        cosines = np.array([0, 0, 1]) --> azimuth=0, dip=-90 (aligned with +ve=up vertical dir)
        cosines = np.array([1, 0, 1]) * 1./np.sqrt(2) --> azimuth=90, dip=-45 deg
        cosines = np.array([-1, 1, -np.sqrt(2)]) * 1./np.sqrt(4) --> azimuth=315, dip=45
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

def convert_pz_to_obspy(pz):
    from obspy.core.inventory.response import PolesZerosResponseStage
    stage_sequence_number = 1
    stage_gain = pz.sensitivity
    stage_gain_frequency = pz.sensitivity_f
    normalization_factor = pz.a0
    normalization_frequency = pz.sensitivity_f

    zeros = pz.zeros
    poles = pz.poles

    if zeros is None:
        print("Inside convert_pz: zeros = None")
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


#obs_Station.x = property(lambda self: self.latitude)

'''
from microquake.core.data.response_utils import read_sensor_types_file, read_cable_file
from instResp.libInst import getResponse, get_corner_freq_from_pole
from instResp.libNom import WA, RC, Accelerometer
from instResp.plotResp import plotResponse
'''

def write_OT_xml(sensor_file, sensor_type_file, cable_file, xml_outfile='OT.xml'):

    if 'SPP_COMMON' not in os.environ:
        print("Set your SPP envs!")
        exit(2)

    path = os.environ['SPP_COMMON'] + '/'

    cables = read_cable_file(path + cable_file)
    sensor_types = read_sensor_types_file(path + sensor_type_file)
    inventory = load_inventory(path + sensor_file)

    for cable in cables:
        print("cable:%s --> %s" % (cable, cables[cable]))
    for sensor in sensor_types:
        print("sensor:%s --> %s" % (sensor, sensor_types[sensor]))

    for station in inventory.stations():

        sensor_type = sensor_types[station.sensor_id]
        cable_cap = cables[station.cable_type]
        '''
        print("sta:%s sensor_id:%s motion:%s output_resist:%s resonant_f0:%s damping:%s" % \
              (station.code, station.sensor_id, station.motion,
               sensor_type['output resistance (ohm)'],
               sensor_type['resonance frequency (Hz)'],
               sensor_type['damping'],
               ))
        '''

        extras = station.extra
        extras['output_resistance_ohm'] = {'namespace': ns, 'value': float(sensor_type['output resistance (ohm)'])}
        extras['resonance_frequency'] = {'namespace': ns, 'value': sensor_type['resonance frequency (Hz)']}
        extras['damping'] = {'namespace': ns, 'value': float(sensor_type['damping'])}
        extras['cable_pF_capacitance_per_m'] = {'namespace': ns, 'value': float(cable_cap)}

        #print("  cable: type:%s len:%s C/m:%s" % (station.cable_type, station.cable_length, cable_cap))

        resistance = float(sensor_type['output resistance (ohm)'])
        if resistance == 0:
            #print("    This is a most likely an Accelerometer!")
            sensitivity = 1.0
        elif resistance % 3500 == 0:
            #print("    This is a high-gain geophone --> Use 80 V/m/s")
            sensitivity = resistance/3500 * 80
        elif resistance % 375 == 0:
            #print("    This is a low-gain geophone --> Use 28.8 V/m/s")
            sensitivity = resistance/375 * 28.8
        else:
            #print("    Unknown resistance [%s] --> use default sensitivity=1.0" % resistance)
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
        freqs = np.logspace(-5, 3., num=500)
        freqs = np.logspace(-5, 4., num=2000)

        pzs.name = station.sensor_id
        pzs.unitsIn = input_units
        pzs.unitsOut = "V"

        if cable_cap == "0":
            #print("No cable cap set --> Skip!")
            #print(pzs)
            pass
        else:
            cable_len = float(station.cable_length)
            # Cable capacity in pF (=10-12 Farads):
            cable_capacity = float(cable_cap) * 1e-12 * cable_len
            tau = resistance * cable_capacity
            f_rc = 1./tau
            #print("cap_per_m:%s x len:%f = %f  x R=%f --> tau=%f fc=1/tau=%g" % \
                  #(cable_cap, cable_len, cable_capacity, resistance, tau, f_rc))
            pz_rc = RC(tau=tau)
            pzs.append_pole(pz_rc.poles[0])
            pzs.normalize_to_a0(norm_freq = 100)


        resp  = getResponse(pzs, freqs, removeZero=False, useSensitivity=False)

        title='sta:%s sensor_type:%s f0=%.0f Hz h=%.2f sensitivity=%.2f' % \
                (station.code, station.sensor_id, f0, damp, sensitivity)
        print("Corner freq:%f" % get_corner_freq_from_pole(pzs.poles[0]))
        if pzs.poles.size == 3:
            print("** High-f Corner freq:%f" % get_corner_freq_from_pole(pzs.poles[2]))

        #if station.code == '2':
        #if 1:
            #plotResponse(resp, freqs, title=title, xmin=1, xmax=10000., ymin=.01, ymax=6, title_font_size=8)
            #exit()

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
            #channel.response.response_stages = stages[0:3]

        '''
        if station.motion == "ACCELERATION":
            response.plot(min_freq=.1, output="ACC")
        '''

    inventory.write(xml_outfile, format='STATIONXML', nsmap={ns_tag: ns})

    return 1


def main():

    #success = write_OT_xml('sensors.csv', 'sensor_types.csv', 'cables.csv', xml_outfile='OT.xml')
    #assert success == 1

    inventory = Inventory.load_from_xml('OT.xml')
    for station in inventory.networks[0].stations:
        print(station.code, station.loc, station.sensor_id, station.extra.damping)
        #for channel in station.channels:
            #output = "VEL"
            #if station.motion == "ACCELERATION":
                #output = "ACC"
            #channel.plot(min_freq=1., output=output)
        #print(station.__dict__)
    exit()

    inventory = load_inventory(os.environ['SPP_COMMON'] + '/sensors.csv')
    sensor_csv = os.environ['SPP_COMMON'] + '/sensors.csv'
    test_read_csv_write_stationxml(sensor_csv, 'OT_new.xml')

    test_read_stationxml('resources/ANMO.xml', 'ANMO2.xml')
    test_read_stationxml('resources/OT.xml', 'OT2.xml')
    test_read_csv_write_stationxml(sensor_csv, 'OT_new.xml')
    test_print_OT_xml_summary('OT_new.xml')


    return



if __name__ == "__main__":
    main()


