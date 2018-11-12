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


def read_csv(filename, site_code='', **kwargs):
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
    from numpy import loadtxt

    data = loadtxt(filename, delimiter=',', skiprows=1, dtype=object)
    stations = []

    for i, tmp in enumerate(data):

        nc, long_name, sc, st, smt, gain, sensitivity = tmp[:7]
        staloc = tmp[7:9]
        orients = tmp[10:22].reshape(3, 4)

        channels = []
        for comp in orients:
            if not comp[0]:
                continue
            xyz = comp[1:4].astype(float)
            channel = Channel(code=comp[0])
            channel.orientation = xyz
            channels.append(channel)

        station = Station(long_name=long_name, code=sc, sensor_type=st,
                          motion_type=smt, gain=gain,
                          sensitivity=sensitivity, loc=staloc,
                          channels=channels)

        stations.append(station)

    networks = [Network(code=nc, stations=stations)]
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
