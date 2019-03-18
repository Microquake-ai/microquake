
""" Collection of functions to calculate moment magnitude

"""

import warnings
import numpy as np
from obspy.core.event.base import Comment, WaveformStreamID
from obspy.core.event.base import ResourceIdentifier
from obspy.core.event.magnitude import Magnitude, StationMagnitude, StationMagnitudeContribution

#from microquake.core.event import (Origin, CreationInfo, Event)
from microquake.waveform.amp_measures import measure_pick_amps
from microquake.waveform.smom_mag import measure_pick_smom
from microquake.waveform.mag_utils import double_couple_rad_pat, free_surface_displacement_amplification

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore")

#1234567890123456789012345678901234567890123456789012345678901234567890123456789

import logging
logger = logging.getLogger(__name__)


def moment_magnitude_new(st, event, vp=5300, vs=3500, ttpath=None,
                         only_triaxial=True, density=2700, min_dist=20,
                         fmin=20, fmax=1000, use_smom=False):

    picks = event.picks
    if use_smom:
        measure_pick_smom(st, picks, debug=False)
        comment = "Average of freq-domain P & S moment magnitudes"
    else:
        measure_pick_amps(st, picks, debug=False)
        comment = "Average of time-domain P & S moment magnitudes"

# Use time(or freq) lambda to calculate moment magnitudes for each arrival

    Mw_P, station_mags_P = calc_magnitudes_from_lambda(cat, vp=vp, vs=vs,
                                                       density=density,
                                                       P_or_S='P',
                                                       use_smom=use_smom)

    Mw_S, station_mags_S = calc_magnitudes_from_lambda(cat, vp=vp, vs=vs,
                                                       density=density,
                                                       P_or_S='S',
                                                       use_smom=use_smom)

# Average Mw_P,Mw_S to get event Mw and wrap with list of
#   station mags/contributions

    Mw = 0.5 * (Mw_P + Mw_S)

    station_mags = station_mags_P + station_mags_S
    set_new_event_mag(event, station_mags, Mw, comment)

    return



def set_new_event_mag(event, station_mags, Mw, comment, make_preferred=False):

    count = len(station_mags)

    sta_mag_contributions = []
    for sta_mag in station_mags:
        sta_mag_contributions.append(StationMagnitudeContribution(
                                      station_magnitude_id=sta_mag.resource_id)
                                    )

    origin_id = event.preferred_origin().resource_id

    event_mag = Magnitude(origin_id=origin_id,
                          mag=Mw,
                          magnitude_type='Mw',
                          station_count=count,
                          evaluation_mode='automatic',
                          station_magnitude_contributions=sta_mag_contributions,
                          comments=[Comment(text=comment)],
                         )

    event.magnitudes.append(event_mag)
    event.station_magnitudes = station_mags

    if make_preferred:
        event.preferred_magnitude_id = ResourceIdentifier(id=event_mag.resource_id.id,
                                                          referred_object=event_mag)

    return


def calc_magnitudes_from_lambda(cat,
                                vp=5300, vs=3500, density=2700,
                                P_or_S='P',
                                use_smom=False,
                                use_sdr_rad=False,
                                use_free_surface_correction=False,
                                min_dist=20.,
                                logger_in=None,
                                **kwargs):
    """
    Calculate the moment magnitude at each station from lambda,
      where lambda is either:
        'dis_pulse_area' (use_smom=False) - calculated by integrating arrival
            displacement pulse in time
        'smom' (use_smom=True) - calculated by fiting Brune spectrum to
            displacement spectrum in frequency
    """

    fname = 'calc_magnitudes_from_lambda'

    global logger

    if logger_in is not None:
        logger = logger_in


# Don't loop over event here, do it in the calling routine
#   so that vp/vs can be set for correct source depth
    event = cat[0]
    origin = event.preferred_origin()
    ev_loc = origin.loc
    origin_id = origin.resource_id

    rad_P, rad_S = 0.52, 0.63

    if P_or_S == 'P':
        v = vp
        rad = rad_P
        mag_type = 'Mw_P'
    else:
        v = vs
        rad = rad_S
        mag_type = 'Mw_S'


    if use_smom:
        magnitude_comment = 'station magnitude measured in frequeny-domain (smom)'
        lambda_key = 'smom'
    else:
        magnitude_comment = 'station magnitude measured in time-domain (dis_pulse_area)'
        lambda_key = 'dis_pulse_area'

    if use_free_surface_correction and np.abs(ev_loc[2]) > 0.:
        logger.warn("%s: Free surface correction requested for event [h=%.1f] > 0" % \
                    (fname, ev_loc[2]))
    if use_sdr_rad and 'sdr' not in kwargs:
        logger.warn("%s: use_sdr_rad requested but NO [sdr] given!" % fname)

    station_mags = []
    Mw_list = []

    Mw_P = []

    arrivals = [arr for arr in event.preferred_origin().arrivals if arr.phase == P_or_S]

    for arr in arrivals:

    #for sta in sorted([sta for sta in st.unique_stations()],
                    #key=lambda x: int(x)):

        pk = arr.pick_id.get_referred_object()
        sta = pk.waveform_id.station_code
        cha = pk.waveform_id.channel_code
        net = pk.waveform_id.network_code

        fs_factor = 1.
        if use_free_surface_correction:
            if arr.get('inc_angle', None):
                inc_angle = arr.inc_angle
                fs_factor = free_surface_displacement_amplification(
                             inc_angle, vp, vs, incident_wave=P_or_S)

            # MTH: Not ready to implement this.  The reflection coefficients
            #      are expressed in x1,x2,x3 coords
                #print("inc_angle:%.1f x1:%.1f x3:%.1f" % (inc_angle, fs_factor[0], fs_factor[2]))
                # MTH: The free surface corrections are returned as <x1,x2,x3>=<
                fs_factor = 1.
            else:
                logger.warn("%s: sta:%s cha:%s pha:%s: inc_angle NOT set in arrival dict --> use default" %\
                            (fname, sta, cha, arr.phase))

        if use_sdr_rad and 'sdr' in kwargs:
            strike, dip, rake = kwargs['sdr']
            if arr.get('takeoff_angle', None) and arr.get('azimuth', None):
                takeoff_angle = arr.takeoff_angle
                takeoff_azimuth = arr.azimuth
                rad = double_couple_rad_pat(takeoff_angle, takeoff_azimuth,
                                            strike, dip, rake, phase=P_or_S)
                rad = np.abs(rad)
                logger.debug("%s: phase=%s rad=%f" % (fname, P_or_S, rad))
                magnitude_comment += ' rad_pat calculated for (s,d,r)=\
                        (%.1f,%.1f,%.1f) theta:%.1f az:%.1f pha:%s |rad|=%f' % \
                        (strike, dip, rake, takeoff_angle, takeoff_azimuth, P_or_S, rad)
                #logger.info(magnitude_comment)
            else:
                logger.warn("%s: sta:%s cha:%s pha:%s: takeoff_angle/azimuth NOT set in arrival dict --> use default radpat" %\
                            (fname, sta, cha, arr.phase))


        _lambda = getattr(arr, lambda_key)

        if _lambda is not None:

            M0_scale = 4. * np.pi * density * v**3 / (rad * fs_factor)

            #R  = np.linalg.norm(sta_dict['station'].loc -ev_loc) #Dist in meters

        # MTH: obspy arrival.distance = *epicentral* distance in degrees
        #   >> Add attribute hypo_dist_in_m to microquake arrival class
        #         to make it clear
            R = arr.hypo_dist_in_m

            if R >= min_dist:

                M0 = M0_scale * R * np.abs(_lambda)
                Mw = 2./3. * np.log10(M0) - 6.033

                Mw_list.append(Mw)

                station_mag = StationMagnitude(origin_id=origin_id, mag=Mw,
                                station_magnitude_type=mag_type,
                                comments=[Comment(text=magnitude_comment)],
                                waveform_id=WaveformStreamID(
                                            network_code=net,
                                            station_code=sta,
                                            channel_code=cha,
                                            ),
                               )
                station_mags.append(station_mag)

            else:
                logger.info("arrival sta:%s pha:%s dist=%.2f < min_dist(=%.2f) --> Skip" % \
                            (fname, sta, arr.phase, R, min_dist))

        #else:
            #logger.warn("arrival sta:%s cha:%s arr pha:%s lambda_key:%s is NOT SET --> Skip" \
                        #% (sta, cha, arr.phase, lambda_key))


    logger.info("nmags=%d avg:%.1f med:%.1f std:%.1f" % \
          (len(Mw_list), np.mean(Mw_list), np.median(Mw_list), np.std(Mw_list)))


    return np.median(Mw_list), station_mags


if __name__ == '__main__':

    main()
