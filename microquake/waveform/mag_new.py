
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore")

from microquake.core.event import (Origin, CreationInfo, Event)

from microquake.core.data.station2 import inv_station_list_to_dict

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

    return


from obspy.core.event.magnitude import Magnitude, StationMagnitude, StationMagnitudeContribution
from obspy.core.event.base import Comment, WaveformStreamID

def calc_magnitudes_from_lambda(st, event, stations, vp=5300, vs=3500, density=2700, P_or_S='P', use_smom=False):
    """
    Calculate the moment magnitude at each station from lambda, where lambda is either:
        'dis_pulse_area' (use_smom=False) - calculated by integrating arrival displacement pulse in time
        'smom' (use_smom=True) - calculated by fiting Brune spectrum to displacement spectrum in frequency
    """

    sta_meta_dict = inv_station_list_to_dict(stations)

    origin = event.preferred_origin()
    ev_loc = origin.loc
    origin_id = origin.resource_id

    rad_P, rad_S = 0.52, 0.63

    if P_or_S == 'P':
        lambda_keys = ['P_arrival']
        v = vp
        rad = rad_P
        mag_type = 'Mw_P'
    else:
        lambda_keys = ['S_arrival']
        v = vs
        rad = rad_S
        mag_type = 'Mw_S'

    magnitude_comment = 'moment magnitude calculated from displacement pulse area ' 
    if use_smom:
        lambda_keys += ['smom']
        magnitude_comment+= 'measured in frequeny-domain (smom)'
    else:
        lambda_keys += ['displacement_pulse', 'dis_pulse_area']
        magnitude_comment+= 'measured in time-domain'


    M0_scale = 4. * np.pi * density * v**3 / rad

    # Loop over unique stations in stream
    # If there are 3 chans and ...
    station_mags = []
    Mw_list = []

    Mw_P = []
    snr_thresh = 8.
    for sta in sorted([sta for sta in st.unique_stations()],
                    key=lambda x: int(x)):

        trs = st.select(station=sta)

        sta_dict = sta_meta_dict[sta]

        nchans_expected = sta_dict['nchans']
        nchans_found = len(trs)

        print("sta:%s nchan_expected:%d nfound:%d" % \
            (sta, nchans_expected, nchans_found))

        # Only process tri-axial sensors (?)
        if nchans_expected != 3:
            continue

        #print("Process sta:%s" % sta)

        vector_sum = 0
        nused = 0

        for i,tr in enumerate(trs):

            has_lambda = True
            _lambda = tr.stats
            for key in lambda_keys:
                try:
                    _lambda = _lambda[key]
                except Exception:
                    print("tr:%s key:%s not found in dict" % (tr.get_id(), key))
                    has_lambda = False
                    break

            #print("tr:%s has_lambda=%s" % (tr.get_id(), has_lambda))

            if has_lambda:

                #snr  = tr.stats[arrival_dict]['snr']
                #print("calc_mag: tr:%s pha:%s snr:%f" % (tr.get_id(), P_or_S, snr))

                vector_sum += _lambda ** 2.
                nused += 1

                """
                if snr >= snr_thresh:
                    vector_sum += tr.stats[arrival_dict][lambda_key] ** 2.
                    nused += 1
                else:
                    print("  Drop tr:%s snr:%.1f < snr_thresh(%.1f)" % (tr.get_id(), snr, snr_thresh))
                """

                R  = np.linalg.norm(sta_dict['station'].loc -ev_loc) # Dist in meters

                M0 = M0_scale * R * np.abs(_lambda)
                equiv_Mw = 2./3. * np.log10(M0) - 6.033
                #Mw_P.append(equiv_Mw)
                snr=1.0
                print("sta:%s cha:%s pha:%s _lambda:%12.10g equiv_Mw:%.2f" % \
                      (sta, tr.stats.channel, P_or_S, _lambda, equiv_Mw))

        if nused > 0:
            vector_sum = np.sqrt(vector_sum)
            R  = np.linalg.norm(sta_dict['station'].loc -ev_loc) # Dist in meters

            M0 = M0_scale * R * vector_sum
            Mw = 2./3. * np.log10(M0) - 6.033
            print("sta:%s [%s] R:%5.1f vector Mw:%5.2f [nused=%d]" % (sta, P_or_S, R, Mw, nused))

            Mw_list.append(Mw)

            station_mag = StationMagnitude(origin_id=origin_id, mag=Mw,
                            station_magnitude_type=mag_type,
                            comments=[Comment(text=magnitude_comment)],
                            waveform_id=WaveformStreamID(
                                        network_code=tr.stats.network,
                                        station_code=tr.stats.station,
                                        channel_code=tr.stats.channel,
                                        ),
                          )
            station_mags.append(station_mag)

    else:
        print(">> Skip 1-C sta:%s" % sta)


    print("nmags=%d avg:%.1f med:%.1f std:%.1f" % \
          (len(Mw_list), np.mean(Mw_list), np.median(Mw_list), np.std(Mw_list)))

    #print("Equiv Mw_%s: nchans=%d mean=%.2f median=%.2f std=%.3f" % (P_or_S, len(Mw_P), np.mean(Mw_P), \
                                                           #np.median(Mw_P), np.std(Mw_P)))

    return np.median(Mw_list), station_mags


if __name__ == '__main__':

    main()
