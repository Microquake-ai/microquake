from os import environ, path

import numpy as np

from microquake.core.data.grid import create, read_grid

from microquake.core.settings import settings


def get_current_velocity_model_id(phase='P'):
    """
    Return the velocity model ID for a specificed phase
    :param phase: phase (possible values 'P', 'S'
    :return: resource_identifier

    """

    if phase.upper() == 'P':
        v_path = path.join(settings.common_dir,
                           settings.grids.velocities.vp) + '.rid'

    elif phase.upper() == 'S':
        v_path = path.join(settings.common_dir,
                           settings.grids.velocities.vs) + '.rid'

    with open(v_path) as ris:
        return ris.read()


def get_velocities():
    """
    returns velocity models
    """

    grids = settings.grids

    if grids.velocities.homogeneous:
        vp = create(**grids)
        vp.data *= grids.velocities.vp
        vp.resource_id = get_current_velocity_model_id('P')
        vs = create(**grids)
        vs.data *= grids.velocities.vs
        vs.resource_id = get_current_velocity_model_id('S')

    else:
        if grids.velocities.source == 'local':
            format = grids.velocities.format
            vp_path = path.join(settings.common_dir,
                                grids.velocities.vp)
            vp = read_grid(vp_path, format=format)
            vp.resource_id = get_current_velocity_model_id('P')
            vs_path = path.join(settings.common_dir,
                                grids.velocities.vs)
            vs = read_grid(vs_path, format=format)
            vs.resource_id = get_current_velocity_model_id('S')
        elif settings['grids.velocities.local']:
            # TODO: read the velocity grids from the server
            pass

    return vp, vs


def create_velocities():

    # Note that this function should not be used forever! New velocity models will be created in the future making
    # this function obsolete

    z = [1168, 459, -300]
    Vp_z = [4533, 5337, 5836]
    Vs_z = [2306, 2885, 3524]

    vp = create(**settings.grids)
    vs = create(**settings.grids)

    origin = settings.grids.origin

    zis = [int(vp.transform_to([origin[0], origin[1], z_])[2]) for z_ in z]

    vp.data[:, :, zis[0]:] = Vp_z[0]
    vs.data[:, :, zis[0]:] = Vs_z[0]

    vp.data[:, :, zis[1]:zis[0]] = np.linspace(Vp_z[1], Vp_z[0], zis[0] - zis[1])
    vs.data[:, :, zis[1]:zis[0]] = np.linspace(Vs_z[1], Vs_z[0], zis[0] - zis[1])

    vp.data[:, :, zis[2]:zis[1]] = np.linspace(Vp_z[2], Vp_z[1], zis[1] - zis[2])
    vs.data[:, :, zis[2]:zis[1]] = np.linspace(Vs_z[2], Vs_z[1], zis[1] - zis[2])

    vp.data[:, :, :zis[2]] = Vp_z[2]
    vs.data[:, :, :zis[2]] = Vs_z[2]

    # TODO this block looks unused to me, should we use it?
    (lx, ly, lz) = vp.shape
    x = [vp.transform_from(np.array([x_, 0, 0]))[0] for x_ in range(0, lx)]
    y = [vp.transform_from(np.array([0, y_, 0]))[1] for y_ in range(0, ly)]
    z = [vp.transform_from(np.array([0, 0, z_]))[2] for z_ in range(0, lz)]

    vp.write(path.join(settings.common_dir, 'velocities/vp'), format='NLLOC')
    vs.write(path.join(settings.common_dir, 'velocities/vs'), format='NLLOC')

    with open(path.join(settings.common_dir, 'velocities/vp.rid'), 'w') as vp:
        vp.write('initial_1d_vp_velocity_model_2018_01')

    with open(path.join(settings.common_dir, 'velocities/vs.rid'), 'w') as vs:
        vs.write('initial_1d_vs_velocity_model_2018_01')
