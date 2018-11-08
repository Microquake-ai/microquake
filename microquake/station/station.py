import obspy.core.inventory
from obspy.core.inventory.util import _unified_content_strings, _textwrap
from obspy.core.inventory.inventory import read_inventory
from obspy.core.inventory import Inventory, Network, Site
from obspy.core import AttribDict
import numpy as np

import csv
import copy

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
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')

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
                station['channels'].append(chan_dict)
                ic += 1

            '''
            print("sta:%3s <%.3f, %.3f> %9.3f %s" % (row['Code'], float(row['Easting']), float(row['Northing']), \
                                                  float(row['Elev']), chans))
            '''
            line_count += 1
            stations.append(station)

    print(f'Processed {line_count} lines.')
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
        if stn.extra:
            sta.extra = copy.deepcopy(stn.extra)
        return sta

    @classmethod
    def from_csv_station(cls, csv_station) -> obspy.core.inventory.Station :
        stn = csv_station

        sta = Station(stn['code'], 0., 0., 0., site=Site(name='dummy site'))
        sta.extra = AttribDict({'x': { 'namespace': ns, 'value': stn['x'], },
                                'y': { 'namespace': ns, 'value': stn['y'], },
                                'z': { 'namespace': ns, 'value': stn['z'], }
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
                                      })
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
        return self._loc


    def __str__(self):
        contents = self.get_contents()
        ret = ("Station {station_name}\n"
               "\tStation Code: {station_code}\n"
               "\tChannel Count: {selected}/{total} (Selected/Total)\n"
               "\t{start_date} - {end_date}\n"
               "\tAccess: {restricted} {alternate_code}{historical_code}\n"
               "\tNorthing: {lat:.2f}, Easting: {lng:.2f}, "
               "Elevation: {elevation:.1f} m\n")
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
            lat=self.y, lng=self.x, elevation=self.z)
        ret += "\tAvailable Channels:\n"
        ret += "\n".join(_textwrap(
            ", ".join(_unified_content_strings(contents["channels"])),
            initial_indent="\t\t", subsequent_indent="\t\t",
            expand_tabs=False))
        return ret

class Channel(obspy.core.inventory.channel.Channel):

    #__doc__ = obsevent.Origin.__doc__.replace('obspy', 'microquake')

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



#obs_Station.x = property(lambda self: self.latitude)
def main():

    stations = read_csv('/Users/mth/mth/python_pkgs/spp/common/sensors.csv')
    obspy_stations = []
    for station in stations:
        obspy_stations.append(Station.from_csv_station(station))

    source = 'mth-test'         # Network ID of the institution sending the message.
    network = Network("OT")
    network.stations = obspy_stations
    inv = Inventory([network], source)
    inv.write('OT3.xml', format='STATIONXML', nsmap={ns_tag: ns})

    inv_read = read_inventory('OT3.xml')
    #inv_read.write('OT4.xml', format='STATIONXML', nsmap={ns_tag: ns})
    for station in inv_read[0].stations:
        stn = Station.from_obspy_station(station)
        print(stn)


if __name__ == "__main__":
    main()
