from obspy.core.event.base import Comment

from loguru import logger
from microquake.waveform.smom_measure import measure_pick_smom

from microquake.core.helpers.grid import synthetic_arrival_times
from microquake.core.settings import settings
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "measure_smom"

    def initializer(self):
        self.use_fixed_fmin_fmax = self.params.use_fixed_fmin_fmax
        self.fmin = self.params.fmin
        self.fmax = self.params.fmax
        self.phase_list = self.params.phase_list

    def process(
        self,
        **kwargs
    ):
        """
        input: catalog, stream

        - origin and picks


        list of corner frequencies for the arrivals
        returns catalog
        """

        logger.info("pipeline: measure smom")

        cat = kwargs["cat"]
        stream = kwargs["stream"]

        plot_fit = False

        missing_responses = stream.attach_response(settings.inventory)

        for sta in missing_responses:
            logger.warning("Inventory: Missing response for sta:%s" % sta)

        for event in cat:
            origin = event.preferred_origin()
            synthetic_picks = synthetic_arrival_times(origin.loc, origin.time)

            for phase in self.phase_list:

                logger.info("Call measure_pick_smom for phase=[%s]" % phase)

                smom_dict, fc = measure_pick_smom(stream, settings.inventory, event,
                                                  synthetic_picks,
                                                  P_or_S=phase,
                                                  fmin=self.fmin, fmax=self.fmax,
                                                  use_fixed_fmin_fmax=self.use_fixed_fmin_fmax,
                                                  plot_fit=plot_fit,
                                                  debug_level=1)
                # except Exception as e:
                #     logger.error(e)
                #     logger.warning("Error in measure_pick_smom. Continuing "
                #                    "to next phase in phase_list: \n %s", e)
                #
                #     continue

                comment = Comment(text="corner_frequency_%s=%.2f measured "
                                       "for %s arrivals" %
                                       (phase, fc, phase))
                cat[0].preferred_origin().comments.append(comment)

        return cat

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        _, stream = self.app.deserialise_message(msg_in)

        return res['cat'], stream
