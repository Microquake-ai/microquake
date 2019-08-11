from microquake.processors.processing_unit import ProcessingUnit
from microquake.core.stream import Stream
from loguru import logger
import numpy as np

from microquake.core.settings import settings

class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "clean_data"

    def process(self, **kwargs):
        """
            Process event and returns its classification.
        """
        waveform = kwargs["waveform"]
        black_list = settings.get('sensors').black_list

        starttime = waveform[0].stats.starttime
        endtime = waveform[0].stats.endtime

        for tr in waveform:
            if tr.stats.starttime < starttime:
                starttime = tr.stats.starttime

            if tr.stats.endtime > endtime:
                endtime = tr.stats.endtime

        waveform.trim(starttime, endtime, pad=True, fill_value=0)

        trs = []

        for i, tr in enumerate(waveform):
            if tr.stats.station not in black_list:
                if np.any(np.isnan(tr.data)):
                    continue
                if ((np.sum(tr.data ** 2) > 0)):
                    trs.append(tr)

        logger.info('The seismograms have  been cleaned, %d trace remaining' %
                    len(trs))

        return Stream(traces=trs)


    # def legacy_pipeline_handler(self, msg_in, res):
    #     """
    #         legacy pipeline handler
    #     """
    #     cat, waveform = self.app.deserialise_message(msg_in)
    #     cat = self.output_catalog(cat)
    #     return cat, waveform
