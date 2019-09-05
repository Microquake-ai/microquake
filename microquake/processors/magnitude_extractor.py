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

        mu = 29.5e9  # rigidity in Pa (shear-wave modulus)

        energy = cat[0].magnitudes[-3].mag
        dict_out['energy_joule'] = energy
        energy_p_dict = eval(cat[0].magnitudes[-3].comments[1].text)
        dict_out['energy_p_joule'] = energy_p_dict['Ep']
        dict_out['energy_p_std'] = energy_p_dict['std_Ep']
        energy_s_dict = eval(cat[0].magnitudes[-3].comments[2].text)
        dict_out['energy_s_joule'] = energy_s_dict['Es']
        dict_out['energy_s_std'] = energy_s_dict['std_Es']

        dict_out['corner_frequency_P_Hz'] = None
        dict_out['corner_frequency_S_Hz'] = None
        cfs = []
        for comment in cat[0].preferred_origin().comments:
            if 'corner_frequency_P' or 'corner_frequency_S' in comment:
                cf_string = cat[0].preferred_origin().comments[0].text
                cf = float(cf_string.split('=')[1].split(' ')[0])
                if '_P' in comment:
                    dict_out['corner_frequency_P_Hz'] = cf
                else:
                    dict_out['corner_frequency_S_Hz'] = cf
                cfs.append(cf)

        cf = np.mean(cfs)
        dict_out['corner_frequency_Hz'] = cf

        td_magnitude = cat[0].magnitudes[-2].mag
        fd_magnitude = cat[0].magnitudes[-1].mag
        dict_out['time_domain_moment_magnitude'] = td_magnitude
        dict_out['frequency_domain_moment_magnitude'] = fd_magnitude
        mw = np.mean([td_magnitude, fd_magnitude])
        dict_out['moment_magnitude'] = mw
        mw_uncertainty = np.abs(td_magnitude - fd_magnitude)
        dict_out['moment_magnitude_uncertainty'] = mw_uncertainty
        sm = 10 ** (3 / 2 * mw + 6.02)
        dict_out['seismic_moment'] = sm
        potency = sm / mu
        dict_out['potency_m**3'] = potency
        dict_out['source_volume_m**3'] = potency
        dict_out['apparent_stress'] = 2 * energy / potency

        ssd = calc_static_stress_drop(mw, cf)
        dict_out['static_stress_drop_MPa'] = ssd
        dict_out['origin_id'] = cat[0].preferred_magnitude().resource_id.id

        return dict_out

