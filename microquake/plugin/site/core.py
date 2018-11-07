# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
#  Purpose: plugin for reading and writing Site object into various format
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
plugin for reading and writing Site object into various format

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from microquake.core import logger

def read_csv(filename, site_code='', has_header=False, **kwargs):
    """
    read a csv file containing sensor information
    The first line of the csv file should contain the site name
    The expected file structure is as follows and should contain one header line
    <network>, <sensor name>, <sensor type>, <no component>, x, y, z
    where x, y and z represents the location of the sensors expressed in a local
    coordinate system. Note that the <sensor name> is limited to four character
    because of NonLinLoc limitations.

    example of file strucuture

    1. <Network>, <sensor long name>, <sensor code>, <sensor type>, <gain>,
    <sensitivity>, <sx>, <sy>, <sz>, <channel 1 code>, <azimuth>, <dip>,
    <channel 2 code>, <azimuth>, <dip>, <channel 3 code>, <azimuth>, <dip>

    :param filename: path to a csv file
    :type filename: string
    :param site_code: site code
    :type site_code: string
    :param has_header: whether or not the input file has an header
    :type has_header: bool
    :rparam: site object
    :rtype: ~microquake.core.station.Site
    """

    from microquake.core.data.station import Site, Network, Station, Channel

    with open(filename) as ifile:
        networks = []
        stations = []
        for i, line in enumerate(ifile.readlines()):
            if has_header and (i == 0):
                continue
            tmp = line.split(',')
            nc = tmp[0]
            long_name = tmp[1]
            sc = tmp[2]
            st = tmp[3]
            smt = tmp[4]
            gain = tmp[5]
            sensitivity = tmp[6]
            sx = float(tmp[7])
            sy = float(tmp[8])
            sz = float(tmp[9])

            channels = []
            for c in range(0, 3):
                cc = tmp[4 * c + 10]
                if not cc:
                    continue
                x = float(tmp[4 * c + 10 + 1])
                y = float(tmp[4 * c + 10 + 2])
                z = float(tmp[4 * c + 10 + 3])
                # az = float(tmp[3 * c + 10 + 1])
                # dip = float(tmp[3 * c + 10 + 2])
                channel = Channel(code=cc)
                channel.orientation = [x, y, z]
                # channel.dip_azimuth = (dip, az)
                channels.append(channel)

            station = Station(long_name=long_name, code=sc, sensor_type=st,
                              motion_type=smt, gain=gain,
                              sensitivity=sensitivity, loc=[sx, sy, sz],
                              channels=channels)

            index = None
            for j, net in enumerate(networks):
                if net.code == nc:
                    index = j

            if index == None:
                network = Network(code=nc, stations=[])
                networks.append(network)
                index = -1

            networks[index].stations.append(station)

    site = Site(code=site_code, networks=networks)
    return site


def read_pickle(filename, **kwargs):
    """
    read site saved pickle format
    :param filename:
    :return:
    """
    from microquake.core.data.station import Site
    import pickle as pickle
    try:
        site = pickle.load(open(filename))
    except:
        logger.error('Not able to read %s' % filename)
        return None

    if not isinstance(site, Site):
        logger.error(
            "The pickle file does not contain and microquake.core.station.Site object")
        return None
    return site


def write_csv(site, filename, **kwargs):
    """
    write a Site object to disk in csv format
    :param filename: full path to file with extension
    :type filename: str
    :param site: site object to be saved
    :type site: ~microquake.core.data.station.Site
    :param protocol: pickling protocol level see pickle.dump documentation
    for more information
    :type protocol: int
    :rtype: None
    """
    # TODO write a function to save the site object in csv format
    pass


def write_pickle(site, filename, protocol=-1, **kwargs):
    """
    write a Site object to disk in pickle (.pickle or .npy extension) format
    using the pickle module
    :param filename: full path to file with extension
    :type filename: str
    :param site: site object to be saved
    :type site: ~microquake.core.data.station.Site
    :param protocol: pickling protocol level see pickle.dump documentation
    for more information
    :type protocol: int
    """
    import pickle as pickle
    with open(filename, 'w') as of:
        pickle.dump(site, of, protocol=protocol)


def write_vtk(site, filename, **kwargs):
    """
    write a Site object to disk in vtk format for viewing in Paraview for
    example
    :param filename: full path to file with extension
    :type filename: str
    :param site: site object to be saved
    :type site: ~microquake.core.data.station.Site
    :param protocol: pickling protocol level see pickle.dump documentation
    for more information
    :type protocol: int
    """
    # TODO write a function to save the site object in vtk format for viewing
    # in paraview
    pass
