import numpy as np

from obspy.core.event.magnitude import Magnitude, StationMagnitude, StationMagnitudeContribution
from obspy.core.event.base import Comment, WaveformStreamID

import matplotlib.pyplot as plt

from microquake.waveform.mag_utils import get_spectra, stack_spectra, calc_fit, calc_fit1, peak_freq

""" This module's docstring summary line.
    This is a multi-line docstring. Paragraphs are separated with blank lines.

    Lines conform to 79-column limit.

    Module and packages names should be short, lower_case_with_underscores.

         1         2         3         4         5         6         7
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""

def measure_pick_smom(st, stations, event, synthetic_picks, fmin=20, fmax=1000, P_or_S='P', debug=False):

# Get P(S) spectra at all stations/channels that have a P(S) arrival:
    sta_dict = get_spectra(st, event, stations, calc_displacement=False, S_win_len=.1, P_or_S=P_or_S)

# Stack vel spectra to get fc ~ peak_f
# Note that fc_S is predicted to be < fc_P 
    stacked_spec, freqs = stack_spectra(sta_dict)
    peak_f = peak_freq(stacked_spec, freqs, fmin=25.)

    plot_spec(stacked_spec, freqs, title='Stack [%s] Vel spec peak_f=%.1f' % (P_or_S, peak_f))

# Now recalculate the spectra as Displacment spectra:

# Get P spectra at all stations/channels that have a P arrival:
    sta_dict = get_spectra(st, event, stations, calc_displacement=True, S_win_len=.1, P_or_S=P_or_S)
    stacked_spec, freqs = stack_spectra(sta_dict)
    fit, fc_stack = calc_fit1(stacked_spec, freqs, fmin=1, fmax=fmax, fit_displacement=True)

    fit,smom_dict = calc_fit(sta_dict, fc=peak_f, fmin=30, fmax=600)

    for tr in st:

        sta = tr.stats.station
        cha = tr.stats.channel

        if sta in smom_dict and cha in smom_dict[sta]:
            smom = smom_dict[sta][cha]['smom']
            print("Found sta:%s cha:%s in smom_dict smom:%12.10g" % (sta, cha, smom))
            pass
        else:
            print("NOT Found sta:%s cha:%s in smom_dict" % (sta, cha))
            continue

        phase = smom_dict[sta][cha]['P_or_S']

        key = "%s_arrival" % phase

        if key in tr.stats:
            #print("Found key=%s in tr.stats --> add to it" % key)
            pass
        else:
            #print("NOT Found key=%s in tr.stats --> create it" % key)
            tr.stats[key] = {}

        tr.stats[key]['smom'] = smom_dict[sta][cha]['smom']
        tr.stats[key]['fit'] = smom_dict[sta][cha]['fit']

    return smom_dict

def moment_magnitude(st, event, stations, vp=5300, vs=3500, ttpath=None, only_triaxial=True, 
                     density=2700, min_dist=20, fmin=20, fmax=1000):
    """
        moment_magnitude - calculate the moment magnitude
    """

    origin_id = event.preferred_origin().resource_id

# Get P spectra at all stations/channels that have a P arrival:
    sta_dict_P = get_spectra(st, event, stations, calc_displacement=False, S_win_len=.1, P_or_S='P')
# Get S spectra at all stations/channels that have an S arrival:
    sta_dict_S = get_spectra(st, event, stations, calc_displacement=False, S_win_len=.1, P_or_S='S')

# Stack vel spectra to get fc ~ peak_f
# Note that fc_S is predicted to be < fc_P 
    stacked_spec_P, freqs = stack_spectra(sta_dict_P)
    stacked_spec_S, freqs_S = stack_spectra(sta_dict_S)

    peak_f = peak_freq(stacked_spec_P, freqs, fmin=25.)
    peak_f_S = peak_freq(stacked_spec_S, freqs, fmin=35.)

    plot_spec(stacked_spec_P, freqs, title='Stack P Vel spec peak_f=%.1f' % peak_f)
    #plot_spec(stacked_spec_S, freqs_S, title='Stack S Vel spec peak_f=%.1f' % peak_f_S)

# Now recalculate the spectra as Displacment spectra:

# Get P spectra at all stations/channels that have a P arrival:
    sta_dict_P = get_spectra(st, event, stations, calc_displacement=True, S_win_len=.1, P_or_S='P')
# Get S spectra at all stations/channels that have an S arrival:
    sta_dict_S = get_spectra(st, event, stations, calc_displacement=True, S_win_len=.1, P_or_S='S')

    stacked_spec_P, freqs = stack_spectra(sta_dict_P)
    fit, fc_stack = calc_fit1(stacked_spec_P, freqs, fmin=1, fmax=fmax, fit_displacement=True)
    stacked_spec_S, freqs_S = stack_spectra(sta_dict_S)
    #fit, fc_stack = calc_fit1(stacked_spec_S, freqs_S, fmin=1, fmax=fmax, fit_displacement=calc_displacement)

    rad_P = 0.52
    rad_S = 0.63

    from time import time
    t1 = time()
    print("Call calc_filt: t1=", t1)

    fit,smom_dict = calc_fit(sta_dict_P, fc=peak_f, fmin=30, fmax=600)
    #fit,smom_dict = calc_fit(sta_dict_P, fc=peak_f, fmin=fmin, fmax=fmax)
    print("******** P smom done, now calc S smom ********")
    fit,smom_dict_S = calc_fit(sta_dict_S, fc=peak_f_S, fmin=30, fmax=600)

    t2 = time()
    print(t2-t1)

    station_mags = []

    Mw_P = []
    Mw_S = []
    Mw_P_wt_avg = Mw_S_wt_avg = 0.
    wts_P = wts_S = 0.

    for some_dict in [smom_dict, smom_dict_S]:
        for sta_code, sta in some_dict.items():
        #for cha_code, smom in sta.items():
            for cha_code, cha_dict in sta.items():

                P_or_S = cha_dict['P_or_S']
                smom = cha_dict['smom']
                fit  = cha_dict['fit']

                print("sta:%s cha:%s R:%.1f [pha:%s] smom:%12.10g" % \
                    (sta_code, cha_code, sta_dict_P[sta_code]['R'], P_or_S, smom))

                lambda_i = smom * sta_dict_P[sta_code]['R']

                if P_or_S == 'P':
                    v = vp
                    rad = rad_P
                elif P_or_S == 'S':
                    v = vs
                    rad = rad_S
                else:
                    print("Ooooops: P_or_S is neither P nor S!")
                    exit()

                scale = 4. * np.pi * density * v**3 / rad
                M0 = lambda_i*scale
                Mw = 2./3. * np.log10(M0) - 6.0333
                print("sta:%s cha:%s smom:%12.10g fit:%.2f [pha:%s] Mw:%.2f" % \
                      (sta_code, cha_code, smom, fit, P_or_S, Mw))

                if P_or_S == 'P':
                    Mw_P.append(Mw)
                    Mw_P_wt_avg += Mw/fit
                    wts_P += 1./fit
                    print("  Add this to Mw_P nP=%d" % len(Mw_P))
                else:
                    Mw_S.append(Mw)
                    Mw_S_wt_avg += Mw/fit
                    wts_S += 1./fit
                    print("  Add this to Mw_S nS=%d" % len(Mw_S))

                mag_type = "Mw_%s" % P_or_S

                station_mag = StationMagnitude(origin_id=origin_id, mag=Mw,
                                     station_magnitude_type=mag_type,
                                     comments=[Comment(text="spectral moment_magnitude")],
                                     waveform_id=WaveformStreamID(network_code="OT",
                                           station_code=sta_code, channel_code=cha_code,),
                              )
                station_mags.append(station_mag)

    Mw_P_wt_avg /= wts_P
    Mw_S_wt_avg /= wts_S
    print("Mw_P mean:%.2f median:%.2f std:%.3f Mw_P_weighted: %.2f" % \
          (np.mean(Mw_P), np.median(Mw_P), np.std(Mw_P), Mw_P_wt_avg))
    print("Mw_S mean:%.2f median:%.2f std:%.3f Mw_S_weighted: %.2f" % \
          (np.mean(Mw_S), np.median(Mw_S), np.std(Mw_S), Mw_S_wt_avg))


    mag = 0.5 * (Mw_P_wt_avg + Mw_S_wt_avg)
    count = len(station_mags)

    sta_mag_contributions = []
    for sta_mag in station_mags:
        sta_mag_contributions.append(StationMagnitudeContribution(station_magnitude_id=sta_mag.resource_id))

    event_mag = Magnitude(origin_id=origin_id,
                          mag=mag,
                          magnitude_type='Mw',
                          station_count=count,
                          evaluation_mode='automatic',
                          station_magnitude_contributions=sta_mag_contributions,
                          comments=[Comment(text="spectral moment_magnitude")],
                          )

    event.magnitudes.append(event_mag)
    event.station_magnitudes = station_mags

    return event


def plot_spec(spec, freqs, title='Stacked spec'):

    plt.loglog(freqs, spec, color='blue')
    plt.xlim(1e0, 3e3)

    if title:
        plt.title(title)
    plt.grid()
    plt.show()


if __name__ == '__main__':

    main()
