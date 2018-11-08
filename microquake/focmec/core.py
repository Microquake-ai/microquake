"""
Package to interact with the FOCMEC sofware (Snoke 2003)
"""

def incidence_phase_vectors(station, origin, velocity=None):
    """
    calculate the incidence vector for the P-, SV- and SH- phase at a station
    for a wave originating from an event given a potentially complex 3D velocity
    model.
    .. NOTE: Straight rays are assumed if no velocity model is provided or if
    the length of the unique value in the velocity model is 1
    :param station: station information
    :type station: ~microquake.core.data.station.Station
    :param velocity: velocity grid
    :type velocity: ~microquake.core.data.grid.GridData
    :param origin: hypocenter location information
    :type origin: ~microquake.core.event.Origin
    :rtype: tuple of P, SV and SH vectors
    """
    from microquake.simul.eik import eikonal_solver
    import numpy as np
    tt = eikonal_solver(velocity, origin.loc, "event")
    # finding the direction of stepest ascent that is the direction of the
    # gradient if the velocity is not homogeneous
    try:
        unique_values = len(np.unique(velocity.data))
    except:
        unique_values = 1

    if velocity or (unique_values > 1):
        tt_gds = np.gradient(tt.data)

        # calculating the incidence vector for the P wave
        P = []
        for tt_gd in tt_gds:
            gd_tmp = tt.copy()
            gd_tmp.data = tt_gd
            P.append(gd_tmp.interpolate(station.loc,
                grid_coordinate=False)[0])

    else:
        # assuming straight ray
        P = station.loc - origin.loc

    # P, SH, SV is a right handed coordinate system
    P = np.array(P)
    P /= np.linalg.norm(P)

    SH = np.array([-P[1], P[0], 0])
    SH /= np.linalg.norm(SH)

    SV = np.cross(P, SH)
    SV /= np.linalg.norm(SV)

    return (P, SV, SH)


def rotate_P_SV_SH(incidence_vects, station):
    """
    rotate into P, SV and SH
    """

def measure_wave_polarity(site, velocity, origin, picks, stream, wave_type='P'):
    """
    calculate the arrival vector at a station for both P-, SV- and SH- waves
    given an event location and a series of picks.
    :param site: information on the station network
    :type site: ~microquake.core.data.station.Site
    :param velocity: P- or S-wave velocity grid
    :type velocity: ~microquake.core.data.grid.GridData
    :param origin: event hypocenter information
    :type origin: ~microquake.core.event.Origin
    :param picks: picks associated to the origin
    :type picks: ~microquake.core.event.Picks
    :param stream: seismogram of the event
    :type stream: ~microquake.core.stream.Stream
    :param wave_type: Phase ('P'. 'SV' or 'SH')
    :type wave_type: str
    """

    # calculate travel time grid from the event location
    from microquake.simul.eik import eikonal_solver

    tt = eikonal_solver(velocity, origin.loc)


def calculate_polarity(station, travel_time, origin, pick):
    """
    Convert first motion polarity as mesured along a given sensor
    component into P or S-wave polarity (depending on the pick phase hint).
    The positive motion direction is defined as the vector moving away from the
    source (compression).

    :param travel_time: travel time grid calculate with the station as seed
    :type travel_time: ~microquake.core.data.grid.GridData
    :param station: station information
    :type station: ~microquake.core.data.station.Station
    :param origin: event origin
    :type origin: ~microquake.core.event.Origin
    :param pick: pick information
    :type pick: ~microquake.core.event.Pick
    :rtype: bool
    """

    # the vector from the station to the event need to be calculated. This will
    # involve calculating a travel time grid from the velocity model.

    pass

