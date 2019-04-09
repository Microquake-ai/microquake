
#import warnings
#warnings.simplefilter("ignore", UserWarning)
#warnings.simplefilter("ignore")

import numpy as np

cos= np.cos
sin= np.sin
degs2rad = np.pi / 180.

def double_couple_rad_pat(takeoff_angle, takeoff_azimuth, strike, dip, rake, phase='P'):
    """
    Return the radiation pattern value at the takeoff point (angle, azimuth) 
        for a specified double couple source
        see Aki & Richards (4.89) - (4.91)
    All input angles in degrees
    allowable phase = ['P', 'SV', 'SH']
    """

    fname = 'double_couple_rad_pat'
    i_h = takeoff_angle * degs2rad
    azd = (takeoff_azimuth - strike) * degs2rad
    # Below is the convention from Lay & Wallace - it looks wrong!
    #azd = (strike - takeoff_azimuth) * degs2rad
    strike = strike * degs2rad
    dip    = dip * degs2rad
    rake   = rake * degs2rad

    radpat = None
    if phase == 'P':
        radpat = cos(rake)*sin(dip)*sin(i_h)**2 * sin(2.*azd)                     \
                -cos(rake)*cos(dip)*sin(2.*i_h) * cos(azd)                        \
                +sin(rake)*sin(2.*dip)*(cos(i_h)**2 - sin(i_h)**2 * sin(azd)**2)  \
                +sin(rake)*cos(2.*dip)*sin(2.*i_h)*sin(azd)

    elif phase == 'SV':
        radpat = sin(rake)*cos(2.*dip)*cos(2.*i_h) * sin(azd)               \
                -cos(rake)*cos(dip)*cos(2.*i_h) * cos(azd)                  \
                +0.5*cos(rake)*sin(dip)*sin(2.*i_h) * sin(2.*azd)           \
                -0.5*sin(rake)*sin(2.*dip)*sin(2.*i_h)*(1 + sin(azd)**2)

    elif phase == 'SH':
        radpat = cos(rake)*cos(dip)*cos(i_h) * sin(azd)                     \
                +cos(rake)*sin(dip)*sin(i_h) * cos(2.*azd)                  \
                +sin(rake)*cos(2.*dip)*cos(i_h) * cos(azd)                  \
                -0.5*sin(rake)*sin(2.*dip)*sin(i_h) * sin(2.*azd)

    elif phase == 'S':
        radpat_SV = sin(rake)*cos(2.*dip)*cos(2.*i_h) * sin(azd)               \
                   -cos(rake)*cos(dip)*cos(2.*i_h) * cos(azd)                  \
                   +0.5*cos(rake)*sin(dip)*sin(2.*i_h) * sin(2.*azd)           \
                   -0.5*sin(rake)*sin(2.*dip)*sin(2.*i_h)*(1 + sin(azd)**2)

        radpat_SH = cos(rake)*cos(dip)*cos(i_h) * sin(azd)                     \
                   +cos(rake)*sin(dip)*sin(i_h) * cos(2.*azd)                  \
                   +sin(rake)*cos(2.*dip)*cos(i_h) * cos(azd)                  \
                   -0.5*sin(rake)*sin(2.*dip)*sin(i_h) * sin(2.*azd)

        radpat = np.sqrt(radpat_SV**2 + radpat_SH**2)

    else:
        print("%s: Unrecognized phase[%s] --> return None" % (fname, phase))
        return None

    return radpat


def free_surface_displacement_amplification(inc_angle, vp, vs, incident_wave='P'):
    """
    Returns free surface displacement amplification for incident P/S wave
        see Aki & Richards prob (5.6)
    All input angles in degrees

    Not sure how useful this will be.
    e.g., It returns the P/SV amplifications for the x1,x3 incidence plane,
    but we rarely rotate into that coord system.
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
        x2_amp = 0.
        # The - is because A&R convention has z-axis positive Down
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


def main():


    vp = 3.
    vs = vp/np.sqrt(3)
    inc_angle = 70.
    ret = free_surface_displacement_amplification(inc_angle, vp, vs, incident_wave='SH')
    assert(ret[1] == 2.)

    vp = 5.
    vs = 3.
    p = 0.2
    inc_angle = np.arcsin(p*vp)
    for inc_angle in range(50): 
        ret = free_surface_displacement_amplification(inc_angle, vp, vs, incident_wave='P')
        print(inc_angle, ret)

    i_h = 45 + 90
    az = 45
    az = -90
    strike=0
    dip=90
    rake=90
    rad_pat = double_couple_rad_pat(i_h, az, strike, dip, rake, phase='P')
    print("i:%f az:%f rad:%f" % (i_h, az, rad_pat))


if __name__ == '__main__':
    main()

