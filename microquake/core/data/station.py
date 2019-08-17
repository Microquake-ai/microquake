import pickle as pickle
from copy import deepcopy

import numpy as np
from pkg_resources import load_entry_point

from microquake.core import logger
from microquake.core.util import ENTRY_POINTS
from microquake.core.util.attribdict import AttribDict


def read_stations(fname, format='CSV', **kwargs):
    """
    Read a site object using plugin
    :param fname: filename
    :param format: one of the supported format
    :return: a site object
    """
    format = format.upper()

    if format not in ENTRY_POINTS['site'].keys():
        logger.error('format %s is not currently supported for Site objects' %
                     format)

        return

    format_ep = ENTRY_POINTS['site'][format]
    read_format = load_entry_point(format_ep.dist.key,
                                   'microquake.plugin.site.%s' % format_ep.name,
                                   'readFormat')

    return read_format(fname, **kwargs)


class Site:
    networks = []

    def __init__(self, code=None, networks=[]):
        self.code = code
        self.networks = networks
        self.__i = 0

    def __setattr__(self, att, value):
        if att == 'networks':
            if isinstance(value, list):
                self.__dict__[att] = value
            else:
                logger.error("stations should be a list of Station")
        else:
            self.__dict__[att] = value

    def __iter__(self):
        return iter(self.networks)

    def next(self):
        if self.__i == len(self.stations):
            self.__i = 0
            raise StopIteration()
        else:
            network = self.networks[self.__i]
            self.__i += 1

            return network

    def write(self, filename, format='PICKLE', **kwargs):
        """
        write the site object to disk
        :param filename: full path to the file to be written
        :type filename: str
        :param format: output file format
        :type format: str
        """
        format = format.upper()

        if format not in ENTRY_POINTS['site'].keys():
            logger.error('format %s is not currently supported for Site '
                         'objects' % format)

            return

        format_ep = ENTRY_POINTS['site'][format]
        write_format = load_entry_point(format_ep.dist.key,
                                        'microquake.plugin.site.%s' % format_ep.name, 'writeFormat')

        write_format(self, filename, **kwargs)

    def copy(self):
        return deepcopy(self)

    def __len__(self):
        return len(self.networks)

    def select(self, network=None, station=None, channel=None,
               sensor_type=None):

        site_tmp = self.copy()
        nets = []

        if network:
            for net in self.networks:
                if net == network:
                    nets.append(net)

            site_tmp.networks = nets

        for inet, net in enumerate(site_tmp.networks):
            site_tmp.networks[inet] = net.select(station=station,
                                                 channel=channel,
                                                 sensor_type=sensor_type)

        return site_tmp

    @property
    def unique_sensor_types(self):
        st = []

        for net in self.networks:
            st.append(net.unique_sensor_types)

        return np.unique(np.array(st).ravel())

    def stations(self, station=None, triaxial=True, uniaxial=True, return_code_only=False):
        """
        :param station: name of a specific station
        :param triaxial: if True return triaxial. Ignored if "station" is
        specified
        :param uniaxial: if True return uniaxial. Ignored if "station" is
        specified
        :return: a list of station
        """
        sts = []
        code = []

        for net in self.networks:
            for sta in net.stations:
                if station:
                    if sta.code == station:
                        sts.append(sta)
                        code.append(sta.code)
                else:
                    if triaxial and len(sta) == 3:
                        sts.append(sta)
                        code.append(sta.code)

                    if uniaxial and len(sta) < 3:
                        sts.append(sta)
                        code.append(sta.code)

        if return_code_only:
            return code
        else:
            return sts


class Network:
    stations = []

    def __init__(self, code=None, stations=[]):
        self.code = code
        self.stations = stations
        self.__i = 0

    def __setattr__(self, att, value):
        if att == 'stations':
            if isinstance(value, list):
                self.__dict__[att] = value
            else:
                logger.error("stations should be a list of Station")
        else:
            self.__dict__[att] = value

    def __len__(self):
        return len(self.stations)

    def __iter__(self):
        return iter(self.stations)

    def next(self):
        if self.__i == len(self.stations):
            self.__i = 0
            raise StopIteration()
        else:
            station = self.stations[self.__i]
            self.__i += 1

            return station

    def copy(self):
        return deepcopy(self)

    def select(self, station=None, sensor_type=None, channel=None):
        stations = []

        if (not station) and (not sensor_type):
            for sta in self.stations:
                stations.append(sta.select(channel=channel))
        elif (station and sensor_type):
            for sta in self.stations:
                if (sta.code == station) and (sta.sensor_type == sensor_type):
                    stations.append(sta.select(channel=channel))
        elif station:
            for sta in self.stations:
                if sta.code == station:
                    stations.append(sta.select(channel=channel))
        elif sensor_type:
            for sta in self.stations:
                if sta.sensor_type == sensor_type:
                    stations.append(sta.select(channel=channel))
        net = self.copy()
        net.stations = stations

        return net

    def unique_sensor_types(self):
        st = []

        for sta in self.stations:
            st.append(sta.sensor_type)

        return np.unique(st)


class Station:
    def __init__(self, long_name=None, code=None, sensor_type=None,
                 motion_type=None, gain=1, sensitivity=1, loc=[0, 0, 0],
                 channels=[]):
        """

        :param long_name: long sensor name
        :param code: sensor code (should be 5 characters or less)
        :param sensor_type: sensor type
        :param motion_type: motion type
        :param gain: analog and digital amplifier gain
        :param sensitivity: sensor sensitivity (e.g. V/g or V/m/s)
        :param loc: sensor location
        :param channels: channels
        :return:
        """
        # inheriting from object won't do anything here as its init is just a pass
        super().__init__()

        self.long_name = long_name
        self.code = code
        self.sensor_type = sensor_type
        self.equipment = AttribDict()
        self.equipment.type = sensor_type
        self.gain = gain
        self.sensitivity = sensitivity
        self.channels = channels
        self.motion_type = motion_type
        self.loc = np.array(loc)
        self.__i = 0

    def __setattr__(self, att, value):
        #print('Station setattr: att=%s --> val=%s' % (att, value))

        if att == 'channels':
            if isinstance(value, list):
                self.__dict__[att] = value
            else:
                raise ValueError("channels should be a list of Channel")
        elif att.lower() == 'x':
            self.__dict__['loc'][0] = value
        elif att.lower() == 'y':
            self.__dict__['loc'][1] = value
        elif att.lower() == 'z':
            self.__dict__['loc'][2] = value
        elif att == "motion_type":
            if value:
                if value.lower() in ["acceleration", "velocity",
                                     "displacement"]:
                    self.__dict__['motion_type'] = value.lower()
                else:
                    logger.warning(
                        'motion_type not set. Possible value are "acceleration", "velocity" and "displacement"')
        elif att == 'loc':
            if isinstance(value, list) or isinstance(value, np.ndarray):
                self.__dict__['loc'] = value
            else:
                logger.error('Not the right type')
        else:
            self.__dict__[att] = value

    def __iter__(self):
        return iter(self.channels)

    def next(self):
        if self.__i == len(self.channels):
            self.__i = 0
            raise StopIteration()
        else:
            channel = self.channels[self.__i]
            self.__i += 1

            return channel

    def __len__(self):
        return len(self.channels)

    @property
    def sensor_type(self):
        return self.equipment.type

    @property
    def x(self):
        return self.loc[0]

    @property
    def X(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]

    @property
    def Y(self):
        return self.loc[1]

    @property
    def z(self):
        return self.loc[2]

    @property
    def Z(self):
        return self.loc[2]

    @property
    def location(self):
        return self.loc

    @property
    def coordinates(self):
        return self.loc

    def copy(self):
        return deepcopy(self)

    def select(self, channel=None):
        channels = []

        if channel:
            for ch in self.channels:
                if ch.code == channel:
                    channels.append(ch)
            sta = self.copy()
            sta.channels = channels

            return sta
        else:
            return self.copy()


class Channel:
    orientation = None

    def __init__(self, code=None, orientation=None):
        self.code = code

        if orientation:
            self.orientation = np.array(orientation) / np.linalg.norm(
                orientation)

    def __setattr__(self, att, value):
        if att == 'code':
            if isinstance(value, str):
                self.__dict__[att] = value.lower()
            else:
                self.__dict__[att] = str(value)
        elif att == 'orientation':
            if (len(value) == 3) and (
                    isinstance(value, list) or isinstance(value, np.ndarray)):
                self.__dict__[att] = value / np.linalg.norm(value)
            else:
                logger.error(
                    'orientation must be a length 3 vector and be type list or a numpy.ndarray')

        elif att.lower() == 'azimuth':
            logger.error(
                "cannot set azimuth directly please set dip_azimuth instead to set dip and azimuth simultaneously")
        # if not np.any(self.orientation):
        # 		logger.error("azimuth cannot when orientation is not defined")
        # 	else:
        # 		hlen = np.linalg.norm(self.orientation[:2])
        # 		dip = np.arctan2(self.orientation[2], hlen) * 180 / np.pi
        # 		az = value * np.pi / 180
        # 		self.__dict__['orientation'] = np.array([np.sin(az) * np.cos(dip), np.cos(az) * np.cos(dip), np.sin(dip)])

        elif att.lower() == 'dip':
            logger.error(
                "cannot set dip directly please set dip_azimuth instead to set dip and azimuth simultaneously")
        # if not np.any(self.orientation):
        # 		logger.error("dip cannot when orientation is not defined")
        # 	else:
        # 		dip = value * np.pi / 180
        # 		az = np.arctan2(self.orientation[0], self.orientation[1]) * 180 / np.pi
        # 		self.__dict__['orientation'] = np.array([np.sin(az) * np.cos(dip), np.cos(az) * np.cos(dip), np.sin(dip)])

        elif att.lower() == 'dip_azimuth':
            if not (len(value) == 2):
                logger.error('input value must be a tupple')
            dip = value[
                0] / 180 * np.pi  # coordinate system z-up dip is define down (need to negate dip to calculate the unit vector)
            az = value[1] / 180 * np.pi
            self.__dict__['orientation'] = np.array(
                [np.sin(az) * np.cos(-dip), np.cos(az) * np.cos(-dip),
                 np.sin(-dip)])

        else:
            self.__dict__[att] = value

    @property
    def azimuth(self):
        try:
            az = np.arctan2(self.orientation[0],
                            self.orientation[1]) * 180 / np.pi
        except:
            az = 0.0

        return az

    @property
    def dip(self):
        try:
            hlen = np.linalg.norm(self.orientation[:2])
            dip = -np.arctan2(self.orientation[2], hlen) * 180 / np.pi
        except:
            dip = 0.0

        return dip

    def copy(self):
        return deepcopy(self)
