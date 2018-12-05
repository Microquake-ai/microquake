import numpy as np

from obspy.core.event.magnitude import Magnitude, StationMagnitude, StationMagnitudeContribution
from obspy.core.event.base import Comment, WaveformStreamID

import matplotlib.pyplot as plt

from microquake.waveform.mag_utils import get_spectra, stack_spectra, calc_fit, calc_fit2, calc_fit3, peak_freq

""" This module's docstring summary line.
    This is a multi-line docstring. Paragraphs are separated with blank lines.

    Lines conform to 79-column limit.

    Module and packages names should be short, lower_case_with_underscores.
"""


def moment_magnitude(st, event, stations, vp=5300, vs=3500, ttpath=None, only_triaxial=True, density=2700,
    min_dist=20, fmin=20, fmax=1000):
    """
        moment_magnitude - calculate the moment magnitude
    """

    origin_id = event.preferred_origin().resource_id

# Get P spectra at all stations/channels that have a P arrival:
    sta_dict_P = get_spectra(st, event, stations, calc_displacement=False, S_win_len=.1, P_or_S='P')
# Get S spectra at all stations/channels that have an S arrival:
    sta_dict_S = get_spectra(st, event, stations, calc_displacement=False, S_win_len=.1, P_or_S='S')

# Stack vel spectra to get fc ~ peak_f
    stacked_spec_P, freqs = stack_spectra(sta_dict_P)
    stacked_spec_S, freqs_S = stack_spectra(sta_dict_S)

# Locate frequency of stack max above f=25 Hz
    peak_f = peak_freq(stacked_spec_P, freqs, fmin=25.)
    title='Stack P spec peak_f=%.1f' % peak_f
    #plot_spec(stacked_spec_P, freqs, title='Stack P spec peak_f=%.1f' % peak_f)

    peak_f_S = peak_freq(stacked_spec_S, freqs_S, fmin=35.)
    #plot_spec(stacked_spec_S, freqs_S, title='Stack S spec peak_f=%.1f' % peak_f_S)

    print("P_df=%8.6f S_df=%8.6f" % (freqs[1], freqs_S[1]))

    rad_P = 0.52
    rad_S = 0.63

    from time import time
    t1 = time()
    print("Call calc_filt: t1=", t1)

    fit,smom_dict = calc_fit(sta_dict_P, fc=peak_f, fmin=fmin, fmax=fmax)
    t2 = time()
    print("Call calc_filt DONE: t2=", t2)
    print(t2-t1)

    station_mags = []

    Mw_P = []
    Mw_P_wt_avg = 0.
    wts = 0.
    for sta_code, sta in smom_dict.items():
        #for cha_code, smom in sta.items():
        for cha_code, cha_dict in sta.items():
            #print("sta:%s cha:%s R:%.1f smom:%12.10g" % (sta_code, cha_code, sta_dict_P[sta_code]['R'],smom))
            smom = cha_dict['smom']
            fit  = cha_dict['fit']

            lambda_i = smom * sta_dict_P[sta_code]['R']
            scale = 4. * np.pi * density * vp**3 / rad_P
            M0 = lambda_i*scale
            Mw = 2./3. * np.log10(M0) - 6.0333
            print("sta:%s cha:%s smom:%12.10g fit:%.2f [P] Mw:%.2f" % (sta_code, cha_code, smom, fit, Mw))
            Mw_P.append(Mw)
            Mw_P_wt_avg += Mw/fit
            wts += 1./fit

            station_mag = StationMagnitude(origin_id=origin_id, mag=Mw,
                                           station_magnitude_type="Mw_P",
                                           comments=[Comment(text="spectral moment_magnitude")],
                                           waveform_id=WaveformStreamID(network_code="OT",
                                                       station_code=sta_code, channel_code=cha_code,),
                                           )
            station_mags.append(station_mag)

    Mw_P_wt_avg /= wts
    print("Mw_P mean:%.2f Mw_P_weighted: %.2f" % (np.mean(Mw_P), Mw_P_wt_avg))

# Use peak_f from P stack or S here ?
    #fit,smom_dict = calc_fit(sta_dict_S, fc=peak_f_S, fmin=fmin, fmax=fmax)
    fit,smom_dict = calc_fit(sta_dict_S, fc=peak_f_S, fmin=1., fmax=fmax)

    Mw_S = []
    Mw_S_wt_avg = 0.
    wts = 0.
    for sta_code, sta in smom_dict.items():
        for cha_code, cha_dict in sta.items():
            #print("sta:%s cha:%s R:%.1f smom:%12.10g" % (sta_code, cha_code, sta_dict_S[sta_code]['R'],smom))
            smom = cha_dict['smom']
            fit  = cha_dict['fit']

            lambda_i = smom * sta_dict_S[sta_code]['R']
            scale = 4. * np.pi * density * vs**3 / (rad_S * 2.)
            M0 = lambda_i*scale
            Mw = 2./3. * np.log10(M0) - 6.0333
            print("sta:%s cha:%s smom:%12.10g [S] Mw:%.2f" % (sta_code, cha_code, smom, Mw))
            Mw_S.append(Mw)
            Mw_S_wt_avg += Mw/fit
            wts += 1./fit

            station_mag = StationMagnitude(origin_id=origin_id,
                                           mag=Mw,
                                           station_magnitude_type="Mw_S",
                                           comments=[Comment(text="spectral moment_magnitude")],
                                           waveform_id=WaveformStreamID(network_code="OT",
                                                       station_code=sta_code, channel_code=cha_code,),
                                           )
            station_mags.append(station_mag)
    Mw_S_wt_avg /= wts
    print("Mw_S mean:%.2f Mw_S_weighted: %.2f" % (np.mean(Mw_S), Mw_S_wt_avg))

    #sta_mag_contributions = [ station_mag ]
    #station_count = 1

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
