
""" Collection of functions to calculate moment magnitude

"""

import warnings
import numpy as np
from obspy.core.event.base import Comment, WaveformStreamID
from obspy.core.event.base import ResourceIdentifier
from obspy.core.event.magnitude import Magnitude, StationMagnitude, StationMagnitudeContribution

#from microquake.core.data.inventory import inv_station_list_to_dict
#from microquake.core.event import (Origin, CreationInfo, Event)
from microquake.waveform.amp_measures import measure_pick_amps
from microquake.waveform.smom_mag import measure_pick_smom
from microquake.waveform.mag_utils import double_couple_rad_pat, free_surface_displacement_amplification

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore")

#1234567890123456789012345678901234567890123456789012345678901234567890123456789

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



def set_new_event_mag(event, station_mags, Mw, comment):

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
    event.preferred_magnitude_id = ResourceIdentifier(id=event_mag.resource_id.id,
                                                      referred_object=event_mag)

    return


def calc_magnitudes_from_lambda(cat,
                                vp=5300, vs=3500, density=2700,
                                P_or_S='P',
                                use_smom=False,
                                use_sdr_rad=False,
                                use_free_surface_correction=False,
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

    #sta_meta_dict = inv_station_list_to_dict(inventory)

# TODO: If you want this function to be a loop over event in cat,
#       need to pass in or calc vp/vs at each source depth
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

    magnitude_comment = 'moment magnitude calculated from displacement pulse area'

    if use_smom:
        magnitude_comment += 'measured in frequeny-domain (smom)'
        lambda_key = 'smom'
    else:
        magnitude_comment += 'measured in time-domain'
        lambda_key = 'dis_pulse_area'


    station_mags = []
    Mw_list = []

    Mw_P = []

    arrivals = [arr for arr in event.preferred_origin().arrivals if arr.phase == P_or_S]

    for arr in arrivals:
    #for arr in event.preferred_origin().arrivals:

    #for sta in sorted([sta for sta in st.unique_stations()],
                    #key=lambda x: int(x)):

        pha = arr.phase

        if pha != P_or_S:
            print("%s: P_or_S=%s but arr pha=%s --> Skip" % (fname, P_or_S, pha))
            continue

        pk = arr.pick_id.get_referred_object()
        sta = pk.waveform_id.station_code
        cha = pk.waveform_id.channel_code
        net = pk.waveform_id.network_code

# TODO: check that inc_angle is set. Also, should we check that this is
#       a surface sensor or just assume it is ??
        fs_factor = 1.
        if use_free_surface_correction:
            inc_angle = arr.inc_angle
            fs_factor = free_surface_displacement_amplification(
                                inc_angle, vp, vs, incident_wave=P_or_S)

        if use_sdr_rad and 'sdr' in kwargs:
            strike, dip, rake = kwargs['sdr']
            takeoff_angle = arr.takeoff_angle
            takeoff_azimuth = arr.azimuth
            rad = double_couple_rad_pat(takeoff_angle, takeoff_azimuth,
                                        strike, dip, rake, phase=P_or_S)
            rad = np.abs(rad)
            magnitude_comment += ' rad_pat calculated for (s,d,r)=\
                    (%.1f,%.1f,%.1f) |rad|=%f' % (strike, dip, rake, rad)


        _lambda = getattr(arr, lambda_key)

        if _lambda is not None:

            M0_scale = 4. * np.pi * density * v**3 / (rad * fs_factor)

            #R  = np.linalg.norm(sta_dict['station'].loc -ev_loc) #Dist in meters

        # MTH: obspy arrival.distance = *epicentral* distance in degrees
        #      So adding attribute hypo_dist_in_m to microquake arrival class
        #      to make it clear
            #R  = arr.distance # Dist in meters
            R = arr.hypo_dist_in_m

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
            print("arrival sta:%s cha:%s arr pha:%s lambda_key:%s is NOT \
                  SET --> Skip" % (sta, cha, pha, lambda_key))


    print("nmags=%d avg:%.1f med:%.1f std:%.1f" % \
          (len(Mw_list), np.mean(Mw_list), np.median(Mw_list), np.std(Mw_list)))

    return np.median(Mw_list), station_mags


if __name__ == '__main__':

    main()
