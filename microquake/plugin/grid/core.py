# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: grid.py
#  Purpose: plugin for reading and writing GridData object into various format 
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
plugin for reading and writing GridData object into various format 

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


def read_nll(filename, **kwargs):
    """
    read NLLoc grid file into a GridData object
    :param filename: filename with or without the extension
    :type filename: str
    :rtype: ~microquake.core.data.grid.GridData
    """
    from microquake.core.nlloc import read_NLL_grid
    return read_NLL_grid(filename)


def read_pickle(filename, **kwargs):
    """
    read grid saved in PICKLE format into a GridData object
    :param filename: full path to the filename
    :type filename: str
    :rtype: ~microquake.core.data.grid.GridData
    """
    import numpy as np
    return np.load(filename)


def read_hdf5(filename, **kwargs):
    """
    read a grid file in hdf5 into a microquake.core.data.grid.GridCollection
    object
    :param filename: filename
    :param kwargs: additional keyword argument passed from wrapper.
    :return: microquake.core.data.grid.GridCollection
    """


def write_nll(grid, filename, **kwargs):
    """
    write a GridData object to disk in NLLoc format
    :param filename: full path to file with or without the extension
    :type filename: str
    :param grid: grid to be saved
    :type grid: ~microquake.core.data.grid.GridData
    """

    from microquake.core.nlloc import write_nll_grid
    data = grid.data
    origin = grid.origin
    spacing = grid.spacing
    grid_type = grid.type
    seed = grid.seed
    label = grid.seed_label
    write_nll_grid(filename, data, origin, spacing, grid_type,
        seed=seed, label=label, **kwargs)



def write_pickle(grid, filename, protocol=-1, **kwargs):
    """
    write a GridData object to disk in pickle (.pickle or .npy extension) format
    using the pickle module
    :param grid: grid to be saved
    :type grid: ~microquake.core.data.grid.GridData
    :param filename: full path to file with extension
    :type filename: str
    :param protocol: pickling protocol level
    :type protocol: int
    """
    import pickle as pickle
    with open(filename, 'wb') as of:
        pickle.dump(grid, of, protocol=protocol)


def write_vtk(grid, filename, *args, **kwargs):
    """
    write a GridData object to disk in VTK format (Paraview, MayaVi2,
    etc.) using
    the pyevtk module.
    param filename: full path to file with the extension. Note that the
    extension for vtk image data (grid data) is usually .vti. 
    :type filename; str
    :param grid: grid to be saved
    :type grid: ~microquake.core.data.grid.GridData
    .. NOTE:
        see the imageToVTK function from the pyevtk.hl module for more
        information on possible additional paramter.
    """
    from pyevtk.hl import imageToVTK

    if filename[-4:] in ['.vti', '.vtk']:
        filename = filename[:-4]

    if isinstance(grid.spacing, tuple):
        spacing = grid.spacing[0]
    else:
        spacing = tuple([grid.spacing] * 3)

    origin = tuple(grid.origin)

    cell_data = {grid.type: grid.data}
    imageToVTK(filename, origin, spacing, pointData = cell_data)


