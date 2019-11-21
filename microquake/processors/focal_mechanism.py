from obspy.core.event.base import ResourceIdentifier

from loguru import logger
from microquake.core.focal_mechanism import calc

from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "focal_mechanism"

    def initializer(self):
        self.save_figs = True

    def process(
        self,
        **kwargs
    ):
        """
        Needs the catalog
        - Origin
        - Picks and Arrivals associated to the Origin

        Returns the focal mechanism
        - list of angles
        - two focal planes
        """

        logger.info("pipeline: focal_mechanism")

        cat = kwargs["cat"].copy()

        focal_mechanisms, figs = calc(cat, self.params)

        if len(focal_mechanisms) > 0:
            for i, event in enumerate(cat):
                focal_mechanism = focal_mechanisms[i]
                cat[i].focal_mechanisms = [focal_mechanism]
                cat[i].preferred_focal_mechanism_id = ResourceIdentifier(
                    id=focal_mechanism.resource_id.id,
                    referred_object=focal_mechanism)
                logger.info(event.preferred_focal_mechanism())

            if self.save_figs:
                for i, fig in enumerate(figs):
                    fig.savefig('foc_mech_%d.png' % i)

        return cat.copy()

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        _, stream = self.app.deserialise_message(msg_in)

        return res['cat'], stream
