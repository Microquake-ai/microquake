import numpy as np

from loguru import logger

from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from microquake.processors.processing_unit import ProcessingUnit
from microquake.core.settings import settings

from microquake.clients.ims.web_client import get_continuous


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "event detection"

    def initializer(self):
        self.sta_lta_on = settings.get('event_detection').sta_lta_on_value
        self.sta_lta_off = settings.get('event_detection').sta_lta_off_value
        self.sta_length_second = settings.get(
            'event_detection').sta_length_second
        self.lta_length_second = settings.get(
            'event_detection').lta_length_second
        self.onset_threshold = settings.get('event_detection').onset_threshold
        self.max_trigger_length_second = settings.get(
            'event_detection').max_trigger_length_second
        self.ims_base_url = settings.get('ims_base_url')
        self.inventory = settings.inventory
        self.settings = settings
        self.network_code = settings.get('network_code')

    def process(
        self,
        **kwargs
    ):
        """
        process(trace)

        scan a trace (not a stream) to detect triggers

        Parameters
        ----------
        trace : a trace object

        Returns
        -------
        catalog: list of trigger on/off times
        """

        logger.info("pipeline: event_detection")

        tr = kwargs['trace']

        tr = tr.detrend('demean').detrend('linear')
        sensor_id = tr.stats.station

        sensor = self.inventory.select(sensor_id)

        poles = np.abs(sensor[0].response.get_paz().poles)
        low_bp_freq = np.min(poles) / (2 * np.pi)
        tr = tr.filter('highpass', freq=low_bp_freq)

        df = tr.stats.sampling_rate
        cft = recursive_sta_lta(tr.data, int(self.sta_length_second * df),
                                int(self.lta_length_second * df))
        on_off = trigger_onset(cft, self.sta_lta_on, self.sta_lta_off)
        sr = tr.stats.sampling_rate
        st = tr.stats.starttime
        on = []
        off = []
        for o_f in on_off:
            on.append(st + o_f[0] / sr)
            off.append(st + o_f[1] / sr)

        return on, off
