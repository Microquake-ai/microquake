
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore")

from microquake.core.event import (Origin, CreationInfo, Event)

from microquake.core.data.inventory import inv_station_list_to_dict

from microquake.waveform.amp_measures import measure_pick_amps
from microquake.waveform.smom_mag import measure_pick_smom

import numpy as np


cos= np.cos
sin= np.sin
degs2rad = np.pi / 180.

def double_couple_rad_pat(takeoff_angle, takeoff_azimuth, strike, dip, rake, phase='P'):
    """
    All input angles in degrees
    allowable phase = ['P', 'SV', 'SH']
    """

    fname = 'double_couple_rad_pat'
    i_h = takeoff_angle * degs2rad
    azd = (strike - takeoff_azimuth) * degs2rad
    strike = strike * degs2rad
    dip    = dip * degs2rad
    rake   = rake * degs2rad

    radpat = -9
    if phase == 'P':
        radpat = cos(strike)*sin(dip)*sin(i_h)**2 * sin(2.*azd)                     \
                -cos(strike)*cos(dip)*sin(2.*i_h) * cos(2.*azd)                     \
                +sin(strike)*sin(2.*dip)*(cos(i_h)**2 - sin(i_h)**2 * sin(azd)**2)  \
                +sin(strike)*cos(2.*dip)*sin(2.*i_h)*sin(azd)

    elif phase == 'SV':
        radpat = sin(strike)*cos(2.*dip)*cos(2.*i_h) * sin(azd)               \
                -cos(strike)*cos(dip)*cos(2.*i_h) * cos(azd)                  \
                +0.5*cos(strike)*sin(dip)*sin(2.*i_h) * sin(2.*azd)           \
                -0.5*sin(strike)*sin(2.*dip)*sin(2.*i_h)*(1 + sin(azd)**2)

    elif phase == 'SH':
        radpat = cos(strike)*cos(dip)*cos(i_h) * sin(azd)                     \
                +cos(strike)*sin(dip)*sin(i_h) * cos(2.*azd)                  \
                +sin(strike)*cos(2.*dip)*cos(i_h) * cos(azd)                  \
                -0.5*sin(strike)*sin(2.*dip)*sin(i_h) * sin(2.*azd)

    else:
        print("%s: Unrecognized phase[%s] --> return None" % (fname, phase))
        return None

    return radpat


def free_surface_displacement_amplification(inc_angle, vp, vs, incident_wave='P'):
    """
    All input angles in degrees
    """

    fname = 'free_surface_displacement_amplification'

    i = inc_angle * degs2rad
    p = sin(i)/vp
    cosi = cos(i)
    cosj = np.sqrt(1. - (vs*p)**2)
    p2= p*p
    b2= vs*vs
    a = (1/b2 - 2.*p2)
    Rpole = a*a + 4.* p2 * cosi/vp * cosj/vs

    if incident_wave == 'P':
        x1_amp = 4.*vp/b2 * p * cosi/vp * cosj/vs / Rpole
        # The - is because A&R convention has z-axis positive Down
        x2_amp = 0.
        x3_amp =-2.*vp/b2 * cosi/vp * a / Rpole

    elif incident_wave == 'SV':
        x1_amp = 2./vs*cosj/vs * a / Rpole
        x2_amp = 0.
        x3_amp = 4./b * p * cosi/vp * cosj/vs / Rpole

    elif incident_wave == 'SH':
        x1_amp = 0.
        x2_amp = 2.
        x3_amp = 0.
    else:
        print("%s: Unrecognized incident wave [%s] --> return None" % (fname, incident_wave))
        return None

    return np.array([x1_amp, x2_amp, x3_amp])

