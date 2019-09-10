from loguru import logger
from microquake.core.stream import Stream
from microquake.waveform.amp_measures import measure_pick_amps
from microquake.waveform.transforms import rotate_to_ENZ, rotate_to_P_SV_SH

from microquake.core.settings import settings
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "measure_amplitudes"

    def process(
        self,
        **kwargs
    ):
        """
        input:
        cat and stream

        obspy modifies the stream in place

        - trace
        - arrivals
        - picks

        then measures amplitude for the first motion

        - adds information to the arrivals

        returns: catalog
        """

        logger.info("pipeline: measure_amplitudes")

        cat = kwargs["cat"]
        stream = kwargs["stream"]

        pulse_min_width = self.params.pulse_min_width
        pulse_min_snr_P = self.params.pulse_min_snr_P
        pulse_min_snr_S = self.params.pulse_min_snr_S
        phase_list = self.params.phase_list

        if not isinstance(phase_list, list):
            phase_list = [phase_list]

        missing_responses = stream.attach_response(settings.inventory)

        for sta in missing_responses:
            logger.warning("Inventory: Missing response for sta:%s" % sta)

        # 1. Rotate traces to ENZ
        st_rot = rotate_to_ENZ(stream, settings.inventory)
        st = st_rot

        # 2. Rotate traces to P,SV,SH wrt event location
        st_new = rotate_to_P_SV_SH(st, cat)
        st = st_new

        # 3. Measure polarities, displacement areas, etc for each pick from
        # instrument deconvolved traces
        trP = [tr for tr in st if tr.stats.channel == 'P' or
               tr.stats.channel.upper() == 'Z']

        measure_pick_amps(Stream(traces=trP),
                          # measure_pick_amps(st_rot,
                          cat,
                          phase_list=phase_list,
                          pulse_min_width=pulse_min_width,
                          pulse_min_snr_P=pulse_min_snr_P,
                          pulse_min_snr_S=pulse_min_snr_S,
                          debug=False)

        return cat.copy()

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        _, stream = self.app.deserialise_message(msg_in)

        return res['cat'], stream
