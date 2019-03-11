import numpy as np

from obspy.core.event.magnitude import Magnitude, StationMagnitude, StationMagnitudeContribution
from obspy.core.event.base import Comment, WaveformStreamID

import matplotlib.pyplot as plt

from microquake.waveform.smom_mag_utils import get_spectra, stack_spectra, calc_fit, calc_fit1, peak_freq

""" This module's docstring summary line.
    This is a multi-line docstring. Paragraphs are separated with blank lines.

    Lines conform to 79-column limit.

    Module and packages names should be short, lower_case_with_underscores.

         1         2         3         4         5         6         7
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""
import logging
logger = logging.getLogger(__name__)

import copy

def measure_pick_smom(st, inventory, event, synthetic_picks,
                      fmin=20, fmax=1000,
                      use_fixed_fmin_fmax=False,
                      plot_fit=False,
                      P_or_S='P',
                      debug_level=0,
                      logger_in=None,
                      **kwargs):

    fname = "measure_pick_smom"

    global logger
    if logger_in is not None:
        logger = logger_in


# Get P(S) spectra at all stations/channels that have a P(S) arrival:
    sta_dict = get_spectra(st, event, inventory, synthetic_picks, calc_displacement=False, S_win_len=.1, P_or_S=P_or_S)

    vel_dict = copy.deepcopy(sta_dict)

# Stack vel spectra to get fc ~ peak_f
# Note that fc_S is predicted to be < fc_P 
    stacked_spec, freqs = stack_spectra(sta_dict)
    peak_f = peak_freq(stacked_spec, freqs, fmin=25.)

    if debug_level > 0:
        logger.debug("%s: pha:%s velocity stack corner freq fc=%.1f" % (fname, P_or_S, peak_f))
    if debug_level > 1:
        plot_spec(stacked_spec, freqs, title='Stack [%s] Vel spec peak_f=%.1f' % (P_or_S, peak_f))

# Now recalculate the spectra as Displacment spectra:

# Get P spectra at all stations/channels that have a P arrival:
    sta_dict = get_spectra(st, event, inventory, synthetic_picks, calc_displacement=True, S_win_len=.1, P_or_S=P_or_S)
    stacked_spec, freqs = stack_spectra(sta_dict)
    fit, fc_stack = calc_fit1(stacked_spec, freqs, fmin=1, fmax=fmax, fit_displacement=True)

    # Calculate fmin/fmax from velocity signal/noise spec
    #   and add into this diplacement spec dict:
    for sta_code, sta_dd in sta_dict.items():
        for cha_code, cha_dict in sta_dd['chan_spec'].items():
            cha_dict['fmin'] = vel_dict[sta_code]['chan_spec'][cha_code]['fmin']
            cha_dict['fmax'] = vel_dict[sta_code]['chan_spec'][cha_code]['fmax']
            #print("sta:%3s cha:%s --> set fmin=%.1f fmax=%.1f" % (sta_code, cha_code, fmin, fmax))

    fit,smom_dict = calc_fit(sta_dict, fc=peak_f, fmin=fmin, fmax=fmax,
                             plot_fit=plot_fit,
                             use_fixed_fmin_fmax=use_fixed_fmin_fmax)

    phase = P_or_S
    arr_dict = {}
    arrivals = [arr for arr in event.preferred_origin().arrivals if arr.phase == phase]
    for arr in arrivals:
        pk = arr.pick_id.get_referred_object()
        sta= pk.waveform_id.station_code
        if sta not in arr_dict:
            arr_dict[sta] = {}
        arr_dict[sta][phase] = arr

    for sta, sta_dict in smom_dict.items():
        smoms=[]
        fits=[]
        for cha, cha_dict in sta_dict.items():
            smoms.append(cha_dict['smom'])
            fits.append(cha_dict['fit'])
            smom = cha_dict['smom']
            fit = cha_dict['fit']

        smom = np.sqrt(np.sum(np.array(smoms)**2))
        fit = np.sum(np.array(fits))/float(len(fits))

        if debug_level > 0:
            logger.debug("%s: sta:%3s pha:%s smom:%12.10g nchans:%d" % (fname, sta, phase, smom, len(fits)))

        arr = arr_dict[sta][phase]
        arr.smom = smom
        arr.fit = fit

    return smom_dict, peak_f


def plot_spec(spec, freqs, title='Stacked spec'):

    plt.loglog(freqs, spec, color='blue')
    plt.xlim(1e0, 3e3)

    if title:
        plt.title(title)
    plt.grid()
    plt.show()


if __name__ == '__main__':

    main()
