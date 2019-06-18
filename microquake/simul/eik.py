# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: eik.py
#  Purpose: Eikonal solver, ray tracer etc. 
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
Eikonal solver, ray tracer etc.

:copyright:
    microquake development team (dev@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


def angles(travel_time):
    """
    This function calculate the take off angle and azimuth for every grid point
    given a travel time grid calculated using an Eikonal solver
    :param travel_time: travel_time grid
    :type travel_time: ~microquake.core.data.grid.GridData with seed property
    (travel_time.seed).
    :rparam: azimuth and takeoff angles grids
    .. Note: The convention for the takeoff angle is that 0 degree is down.
    """
    import numpy as np

    gds_tmp = np.gradient(travel_time.data)
    gds = [-gd for gd in gds_tmp]

    tmp = np.arctan2(gds[0], gds[1])  # azimuth is zero northwards
    azimuth = travel_time.copy()
    azimuth.type = 'ANGLE'
    azimuth.data = tmp
    if len(travel_time.shape) == 3:
        hor = np.sqrt(gds[0] ** 2 + gds[1] ** 2)
        tmp = np.arctan2(hor, -gds[2])
        # takeoff is zero pointing down
        takeoff = travel_time.copy()
        takeoff.type = 'ANGLE'
        takeoff.data = tmp
        return azimuth, takeoff
    else:
        return azimuth


def ray_tracer(travel_time, start, grid_coordinates=False, max_iter=1000):
    """
    This function calculates the ray between a starting point (start) and an end
    point, which should be the seed of the travel_time grid, using the gradient
    descent method. 
    :param trave_time: travel time grid with a seed defined
    :type travel_time: ~microquake.core.data.grid.GridData with an additional
    seed property(travel_time.seed). Note that seed is automatically added to
    the travel time grid by the Eikonal solver or when read from NLLOC grid
    file.
    :param start: the starting point (usually event location)
    :type start: tuple, list or numpy.array
    :param grid_coordinates: if true grid coordinates (indices,
    not necessarily integer are used, else real world coordinates are used
    (x, y, z) (Default value False)
    :param max_iter: maximum number of iteration
    :rtype: numpy.array
    """

    import numpy as np
    from microquake.core import GridData
    from microquake.core.event import Ray

    if grid_coordinates:
        start = np.array(start)
        start = travel_time.transform_from(start)

    origin = travel_time.origin
    spacing = travel_time.spacing
    end = np.array(travel_time.seed)
    start = np.array(start)

    # calculating the gradient in every dimension at every grid points
    gds_tmp = np.gradient(travel_time.data)
    gds = [GridData(gd, origin=origin, spacing=spacing,) for gd in gds_tmp]

    dist = np.linalg.norm(start - end)
    cloc = start  # initializing cloc "current location" to start
    gamma = spacing / 2    # gamma is set to half the grid spacing. This
                         # should be
                         # sufficient. Note that gamma is fixed to reduce
                         # processing time.
    nodes = [start]
    while dist > spacing / 2:

        if dist < spacing * 4:
            gamma = spacing / 4

        gvect = np.array([gd.interpolate(cloc, grid_coordinate=False,
                          order=1)[0] for gd in gds])


        cloc = cloc - gamma * gvect / np.linalg.norm(gvect)
        nodes.append(cloc)
        dist = np.linalg.norm(cloc - end)

    nodes.append(end)

    ray = Ray(nodes=nodes)
    return ray


def eikonal_solver(velocity, seed, seed_label, *args, **kwargs):
    """
    Eikonal solver based of scikit fast marching solver interfaced for
    microquake
    :param velocity: velocity grid
    :type velocity: ~microquake.core.data.grid.GridData
    :param seed: numpy array location of the seed or origin of seismic wave in model coordinates
    (usually location of a station or an event)
    :type seed: numpy array
    :param seed_label: seed label (name of station)
    :type seed_label: basestring
    """


    import skfmm
    import numpy as np
    seed = np.array(seed)

    phi = -1*np.ones_like(velocity.data)
    seed_coord = velocity.transform_to(seed)

    phi[tuple(seed_coord.astype(int))] = 1
    tt = skfmm.travel_time(phi, velocity.data, dx=velocity.spacing, *args,
            **kwargs)
    tt_grid = velocity.copy()
    tt_grid.data = tt
    tt_grid.seed = seed
    tt_grid.seed_label = seed_label

    return tt_grid


def sensitivity_location(velocity, seed, location, perturbation=0.1, h=1):
    """
    Calculate the sensitivity kernel for location in seed
    :param velocity: a velocity grid
    :type velocity: microquake.core.data.GridData
    :param seed: seed for traveltime grid
    :type seed: numpy.array
    :param location: location at which the sensitivity is evaluated
    :type location: numpy.array
    :param perturbation: perturbation to the location in the same using as loc (m, km etc)
    :type perturbation: float
    :param h:
    :rparam: location sensitivity at the provided location
    :rtype: numpy.array
    """

    # creating a buffer around velocity in all dimensions
    # works only in 3D ...

    import numpy as np
    from scipy.ndimage.interpolation import map_coordinates

    buf = 2


    x = np.arange(-buf, velocity.data.shape[0] + buf)
    y = np.arange(-buf, velocity.data.shape[1] + buf)
    z = np.arange(-buf, velocity.data.shape[2] + buf)

    Y, X, Z = np.meshgrid(y, x, z)
    X1 = X.ravel()
    Y1 = Y.ravel()
    Z1 = Z.ravel()

    coords = np.vstack((X1, Y1, Z1))
    vel = velocity.copy()

    vel.data = map_coordinates(velocity.data, coords, mode='nearest').reshape(X.shape)

    traveltime = eikonal_solver(vel, seed)

    h = float(h)

    ndim = len(traveltime.shape)

    spc = traveltime.spacing
    shape = np.array(traveltime.shape)

    frechet = []
    end = traveltime.transform_to(location) + buf
    for j in range(len(seed)):
        new_end1 = end.copy()

        new_end1[(end[:, j] + perturbation < shape[j]) & (end[:, j] + perturbation > 0), j] += perturbation

        new_end2 = end.copy()
        new_end2[(end[:, j] - perturbation < shape[j]) & (end[:, j] - perturbation > 0), j] -= perturbation

        perturbated_tt1 = map_coordinates(traveltime.data, new_end1.T , order=1, mode='nearest')
        perturbated_tt2 = map_coordinates(traveltime.data, new_end2.T, order=1, mode='nearest')

        f = (perturbated_tt1 - perturbated_tt2) / ((new_end1[:, j] - new_end2[:, j]) * spc)

        frechet.append(f)

    frechet = np.array(frechet)

    return frechet.T


def sensitivity_velocity(velocity, seed, start_points, perturbation=0.1, h=1):
    """
    Calculate the sensitivity kernel (Frechet derivative, dt/dV)
    for every velocity element (v_i)

    The sensitivity is calculated as follows for all velocity element i:

    dt/dv_i = l_i * (v_i+ - v_i-) / (2 * perturbation), where

    l_i =  T_i / v_i,  v_i+ and v_i- are is v_i +/- perturbation, respectively.
    T represents the travel time grid calculated using the eikonal_solver

    :param velocity: velocity grid
    :type velocity: microquake.core.data.GridData
    :param seed: seed for traveltime grid (usually sensor location)
    :type seed: numpy.array
    :param start_points: start points for the ray tracing, usually event location
    :type start_points: numpy.array coordinates
    :param time: time grid
    :type time: microquake.core.data.GridData
    :param perturbation: velocity perturbation
    :param h:
    :return: sensitivity kernel for velocity
    """

    import numpy as np
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.sparse import csr_matrix

    # initiating the frechet derivative matrix
    n_node = np.prod(velocity.shape)
    n_measurement = len(start_points)
    F = csr_matrix((n_measurement, n_node), dtype=np.float32)

    # adding buffer to the velocity
    buf = 2 # related to the use of cubic spline with

    csr_matrix

    x = []
    for dim in range(len(velocity.data.shape)):
        x.append(np.arange(-buf, velocity.data.shape[0] + buf))

    if len(velocity.data.shape) == 3:
        X, Y, Z = np.meshgrid(x[0], x[1], x[2])
        X1 = X.ravel()
        Y1 = Y.ravel()
        Z1 = Z.ravel()
        coords = np.vstack((X1, Y1, Z1))
    else:
        X, Y = np.meshgrid(x[0], x[1], x[2])
        X1 = X.ravel()
        Y1 = Y.ravel()
        coords = np.vstack((X1, Y1))

    vel = velocity.copy()
    vel.data = map_coordinates(velocity.data, coords, mode='nearest').reshape(X.shape)

    #travel_time = EikonalSolver(vel, seed)
    for start in start_points:
        ray = ray_tracer(travel_time, start)
        for segment in ray:
            pass