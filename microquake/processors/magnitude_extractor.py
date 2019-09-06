from loguru import logger
import numpy as np

# from microquake.core.settings import settings
from microquake.processors.processing_unit import ProcessingUnit
from microquake.waveform.mag_utils import calc_static_stress_drop


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "extract_magnitude"

    def process(
            self,
            **kwargs
    ):
        logger.info("pipeline: measure_amplitudes")

        cat = kwargs["cat"]

        dict_out = {}
        dict_keys = ['energy_joule', 'energy_p_joule', 'energy_p_std',
                     'energy_s_joule', 'energy_s_std',
                     'corner_frequency_p_hz', 'corner_frequency_s_hz',
                     'time_domain_moment_magnitude',
                     'frequency_domain_moment_magnitude',
                     'moment_magnitude', 'moment_magnitude_uncertainty',
                     'seismic_moment', 'potency_m3', 'source_volume_m3',
                     'apparent_stress', 'static_stress_drop_mpa', 'origin_id']

        for key in dict_keys:
            dict_out[key] = None


        mu = 29.5e9  # rigidity in Pa (shear-wave modulus)

        # finding the index for magnitude object that contains the energy

        for magnitude in reversed(cat[0].magnitudes):

            if magnitude.magnitude_type == 'E':
                energy = magnitude.mag
                dict_out['energy_joule'] = energy
                energy_p_dict = eval(magnitude.comments[1].text)
                dict_out['energy_p_joule'] = energy_p_dict['Ep']
                dict_out['energy_p_std'] = energy_p_dict['std_Ep']
                energy_s_dict = eval(magnitude.comments[2].text)
                dict_out['energy_s_joule'] = energy_s_dict['Es']
                dict_out['energy_s_std'] = energy_s_dict['std_Es']

                break

        for magnitude in reversed(cat[0].magnitudes):

            if 'time domain station magnitudes' in magnitude.comments[0]:
                td_magnitude = magnitude
                dict_out['time_domain_moment_magnitude'] = td_magnitude

                break

        for magnitude in reversed(cat[0].magnitudes):

            if 'frequency domain station magnitudes' in magnitude.comments[0]:
                fd_magnitude = magnitude
                dict_out['frequency_domain_moment_magnitude'] = fd_magnitude

                break

        dict_out['corner_frequency_p_hz'] = None
        dict_out['corner_frequency_s_hz'] = None
        cfs = []
        for comment in cat[0].preferred_origin().comments:
            if 'corner_frequency_p' or 'corner_frequency_s' in comment:
                cf_string = cat[0].preferred_origin().comments[0].text
                cf = float(cf_string.split('=')[1].split(' ')[0])
                if '_P' in comment:
                    dict_out['corner_frequency_p_hz'] = cf
                else:
                    dict_out['corner_frequency_s_hz'] = cf
                cfs.append(cf)

        cf = np.mean(cfs)
        dict_out['corner_frequency_hz'] = cf

        if td_magnitude is None and fd_magnitude is None:
            mw = None
            mw_uncertainty = None

        elif td_magnitude:
            mw = fd_magnitude
            mw_uncertainty = None

        elif fd_magnitude is None:
            mw = td_magnitude
            mw_uncertainty = None

        else:
            mw = np.mean([td_magnitude, fd_magnitude])
            mw_uncertainty = np.abs(td_magnitude - fd_magnitude)

        dict_out['moment_magnitude'] = mw
        dict_out['moment_magnitude_uncertainty'] = mw_uncertainty
        sm = 10 ** (3 / 2 * mw + 6.02)
        dict_out['seismic_moment'] = sm
        potency = sm / mu
        dict_out['potency_m3'] = potency
        dict_out['source_volume_m3'] = potency
        dict_out['apparent_stress'] = 2 * energy / potency

        ssd = calc_static_stress_drop(mw, cf)
        dict_out['static_stress_drop_mpa'] = ssd
        dict_out['origin_id'] = cat[0].preferred_magnitude().resource_id.id

        return dict_out

