from loguru import logger
from microquake.waveform.amp_measures import calc_velocity_flux
from microquake.waveform.mag import calculate_energy_from_flux

from microquake.processors.processing_unit import ProcessingUnit

from microquake.core.focal_mechanism import calc

from microquake.waveform.simple_mag import moment_magnitude
from microquake.waveform.amp_measures import measure_pick_amps
from microquake.core.helpers import velocity


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "simple_magnitude"

    def process(
        self,
        **kwargs
    ):

        stream = kwargs['stream']
        cat = kwargs['cat']

        inventory = self.settings.inventory
        stream.attach_response(inventory)

        vp, vs = velocity.get_velocities()
        cat_moment = moment_magnitude(stream, cat, inventory, vp, vs)

        # cat_pick_amps = measure_pick_amps(stream, cat_moment, phase_list=['P',
        #                                                                  'S'])
        # focal_mechanisms, _ = calc(cat_pick_amps,
        #                            self.settings.get('focal_mechanism'))

        # cat_focal_mechanisms = cat_pick_amp[0].copy()
        # cat_focal_mechanisms.focal_mechanisms = focal_mechanisms

        cat_flux = calc_velocity_flux(stream, cat_moment, inventory,
                                      phase_list=['P', 'S'])
        cat_energy = calculate_energy_from_flux(cat_flux, inventory, vp, vs,
                                                use_sdr_rad=False)

        mag = cat_energy[0].magnitudes[-1]
        energy_p = mag.energy_p_joule
        energy_s = mag.energy_s_joule
        energy = mag.energy_joule
        cat_energy[0].preferred_magnitude().energy_p_joule = energy_p
        cat_energy[0].preferred_magnitude().energy_s_joule = energy_s
        cat_energy[0].preferred_magnitude().energy_joule = energy

        return cat_energy

