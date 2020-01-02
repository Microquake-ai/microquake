import matplotlib.pyplot as plt
import mplstereonet
import numpy as np

from loguru import logger


def plot_beachball(cat):

    if cat[0].preferred_focal_mechanism is None:
        logger.warning('nothing to do, the catalog does not have a '
                       'preferred focal mechanism')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')

    # plotting the nodal planes

    np1 = cat[0].preferred_focal_mechanism().nodal_planes.nodal_plane_1
    np2 = cat[0].preferred_focal_mechanism().nodal_planes.nodal_plane_2
    s1 = np1['strike']
    d1 = np1['dip']
    s2 = np2['strike']
    d2 = np2['dip']

    ax.plane(s1, d1, 'k')
    ax.plane(s2, d2, 'k')

    takeoffs = [ar.takeoff_angle for ar in cat[0].preferred_origin().arrivals]
    azimuths = [ar.azimuth for ar in cat[0].preferred_origin().arrivals]

    takeoffs = np.array(takeoffs)
    azimuths = np.array(azimuths)

    polarities = [ar.polarity for ar in cat[0].preferred_origin().arrivals]
    polarities = np.array(polarities, dtype=np.float)

    up = polarities > 0
    dn = polarities < 0

    # compression
    h_rk = ax.line(90. - takeoffs[up], azimuths[up], 'ko')
    # first arrivals
    h_rk = ax.line(90. - takeoffs[dn], azimuths[dn], 'ko', fillstyle='none')

