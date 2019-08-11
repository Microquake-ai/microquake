from numpy import array

from microquake.core.util.attribdict import AttribDict

from microquake.core.settings import settings
from microquake.core.helpers.velocity import get_velocities


def nll_velgrids():
    """
    Returns the information required by nll to initialize the nll object
    Returns:

    """

    out_dict = AttribDict()

    vp, vs = get_velocities()

    out_dict = AttribDict()
    out_dict.vp = settings.grids.velocities.vp
    out_dict.vs = settings.grids.velocities.vs
    out_dict.homogeneous = \
        settings.grids.velocities.homogeneous
    out_dict.grids = AttribDict()
    out_dict.grids.vp = vp
    out_dict.grids.vs = vs

    out_dict.index = 0

    return out_dict
    # reading the station information


def nll_sensors():
    """
    Returns the information required by nll to initialize the nll object
    Returns: AttribDict

    """

    out_dict = AttribDict()

    # TODO: fix / type annotation
    stations = settings.inventory.stations()
    out_dict.name = array([station.code for station in stations])
    out_dict.pos = array([station.loc for station in stations])
    out_dict.site = "THIS IS NOT SET"

    '''
    site = self.get_stations()
    out_dict.site = site
    out_dict.name = array([station.code for station in site.stations()])
    out_dict.pos = array([station.loc for station in site.stations()])
    '''
    out_dict.key = '0'
    out_dict.index = 0

    return out_dict
