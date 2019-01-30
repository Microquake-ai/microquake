
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore")

from microquake.core.event import (Origin, CreationInfo, Event)

from microquake.core.data.inventory import inv_station_list_to_dict

from microquake.waveform.amp_measures import measure_pick_amps
from microquake.waveform.smom_mag import measure_pick_smom

import numpy as np


def moment_magnitude_new(st, event, stations, vp=5300, vs=3500, ttpath=None, only_triaxial=True,
                         density=2700, min_dist=20, fmin=20, fmax=1000, use_smom=False):

    picks = event.picks
    if use_smom:
        measure_pick_smom(st, picks, debug=False)
        comment="Average of freq-domain P & S moment magnitudes"
    else:
        measure_pick_amps(st, picks, debug=False)
        comment="Average of time-domain P & S moment magnitudes"

# Use time(or freq) lambda to calculate moment magnitudes for each arrival

    Mw_P, station_mags_P = calc_magnitudes_from_lambda(st, event, stations, vp=vp, vs=vs,
                                                       density=2700, P_or_S='P', use_smom=use_smom)

    Mw_S, station_mags_S = calc_magnitudes_from_lambda(st, event, stations, vp=vp, vs=vs,
                                                       density=2700, P_or_S='S', use_smom=use_smom)

# Average Mw_P,Mw_S to get event Mw and wrap with list of station mags/contributions

    Mw = 0.5 * (Mw_P + Mw_S)

    station_mags = station_mags_P + station_mags_S
    set_new_event_mag(event, station_mags, Mw, comment)

    return



from obspy.core.event.base import ResourceIdentifier
def set_new_event_mag(event, station_mags, Mw, comment):

    count = len(station_mags)

    sta_mag_contributions = []
    for sta_mag in station_mags:
        sta_mag_contributions.append( StationMagnitudeContribution(
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
    event.preferred_magnitude_id = ResourceIdentifier(id=event_mag.resource_id.id, referred_object=event_mag)

    return


from obspy.core.event.magnitude import Magnitude, StationMagnitude, StationMagnitudeContribution
from obspy.core.event.base import Comment, WaveformStreamID

#def calc_magnitudes_from_lambda(st, event, stations, vp=5300, vs=3500, density=2700, P_or_S='P', use_smom=False):
def calc_magnitudes_from_lambda(cat, inventory, vp=5300, vs=3500, density=2700, P_or_S='P', use_smom=False):
    """
    Calculate the moment magnitude at each station from lambda, where lambda is either:
        'dis_pulse_area' (use_smom=False) - calculated by integrating arrival displacement pulse in time
        'smom' (use_smom=True) - calculated by fiting Brune spectrum to displacement spectrum in frequency
    """

    sta_meta_dict = inv_station_list_to_dict(inventory)
    #sta_meta_dict = inv_station_list_to_dict(stations)

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

    magnitude_comment = 'moment magnitude calculated from displacement pulse area ' 

    if use_smom:
        magnitude_comment+= 'measured in frequeny-domain (smom)'
        lambda_key = 'smom'
    else:
        magnitude_comment+= 'measured in time-domain'
        lambda_key = 'dis_pulse_area'


    M0_scale = 4. * np.pi * density * v**3 / rad

    # Loop over unique stations in stream
    # If there are 3 chans and ...
    station_mags = []
    Mw_list = []

    Mw_P = []

    for arr in event.preferred_origin().arrivals:

    #for sta in sorted([sta for sta in st.unique_stations()],
                    #key=lambda x: int(x)):

        pha = arr.phase

        pk = arr.pick_id.get_referred_object()
        sta = pk.waveform_id.station_code
        cha = pk.waveform_id.channel_code
        net = pk.waveform_id.network_code

        sta_dict = sta_meta_dict[sta]

        print("calc_magnitudes: sta:%s cha:%s pha:%s" % (sta, cha, pha))

        #if arr.dis_pulse_area is not None: 
        _lambda = getattr(arr, lambda_key)

        if _lambda is not None: 

            #_lambda = arr.dis_pulse_area

            R  = np.linalg.norm(sta_dict['station'].loc -ev_loc) # Dist in meters

            M0 = M0_scale * R * np.abs(_lambda)
            Mw = 2./3. * np.log10(M0) - 6.033
            #print("sta:%s cha:%s pha:%s _lambda:%12.10g equiv_Mw:%.2f" % \
                  #(sta, tr.stats.channel, P_or_S, _lambda, equiv_Mw))

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
            print("sta:%s cha:%s arr pha:%s lambda_key:%s is NOT SET --> Skip" % \
                  (sta, cha, pha, lambda_key))


    print("nmags=%d avg:%.1f med:%.1f std:%.1f" % \
          (len(Mw_list), np.mean(Mw_list), np.median(Mw_list), np.std(Mw_list)))

    #print("Equiv Mw_%s: nchans=%d mean=%.2f median=%.2f std=%.3f" % (P_or_S, len(Mw_P), np.mean(Mw_P), \
                                                           #np.median(Mw_P), np.std(Mw_P)))

    return np.median(Mw_list), station_mags


if __name__ == '__main__':

    main()
