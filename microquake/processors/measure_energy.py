from loguru import logger
from microquake.waveform.amp_measures import calc_velocity_flux
from microquake.waveform.mag import calculate_energy_from_flux

from microquake.core.settings import settings
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "measure_energy"

    def process(
        self,
        **kwargs
    ):
        logger.info("pipeline: measure energy")

        cat = kwargs["cat"]
        stream = kwargs["stream"]

        correct_attenuation = self.params.correct_attenuation
        Q = self.params.attenuation_Q
        use_sdr_rad = self.params.use_sdr_rad

        if use_sdr_rad and cat.preferred_focal_mechanism() is None:
            logger.warning("use_sdr_rad=True but preferred focal mech = None --> Setting use_sdr_rad=False")
            use_sdr_rad = False

        phase_list = self.params.phase_list

        if not isinstance(phase_list, list):
            phase_list = [phase_list]

        missing_responses = stream.attach_response(settings.inventory)

        for sta in missing_responses:
            logger.warning("Inventory: Missing response for sta:%s" % sta)

        missing_responses_ids = {r.id for r in missing_responses}
        cleaned_stream = stream.copy()
        cleaned_stream.traces = [tr for tr in cleaned_stream.traces
                                 if tr.id not in missing_responses_ids]

        calc_velocity_flux(cleaned_stream,
                           cat,
                           phase_list=phase_list,
                           correct_attenuation=correct_attenuation,
                           Q=Q,
                           debug=False)

        calculate_energy_from_flux(cat,
                                   use_sdr_rad=use_sdr_rad)

        self.result = {'cat': cat}
        return self.result

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        _, stream = self.app.deserialise_message(msg_in)

        return res['cat'], stream
