from microquake.ml.classifier import SeismicModel
from microquake.processors.processing_unit import ProcessingUnit
from loguru import logger


class Processor(ProcessingUnit):
    """
        Class wrapper around SeismicClassifierModel, load inputs from kwargs.
    """
    @property
    def module_name(self):
        return "event_classifier"

    def initializer(self):
        self.seismic_model = SeismicModel()
        self.seismic_model.create_model()

    def process(self, **kwargs):
        """
            Process event and returns its classification.
        """
        stream = kwargs["stream"]
        cat = kwargs["cat"]
        context_trace = kwargs["context"].composite()
        station = context_trace[0].stats.station

        tr = stream.select(station=station).composite()

        if cat[0].preferred_origin() is None:
            logger.warning('the catalog preferred origin is None. Using the '
                           'last inserted origin')
            elevation = cat[0].origins[-1].z
        else:
            elevation = cat[0].preferred_origin().z

        if cat[0].preferred_magnitude() is None:
            logger.warning('the catalog preferred magnitude is None. Using '
                           'the last inserted magnitude')
            magnitude = cat[0].magnitudes[-1].mag

        else:
            magnitude = cat[0].preferred_magnitude().mag



        response = self.seismic_model.predict(tr, context_trace,
                                              elevation, magnitude)
        return response

    def legacy_pipeline_handler(self, msg_in, res):
        """
            legacy pipeline handler
        """
        cat, stream = self.app.deserialise_message(msg_in)
        cat = self.output_catalog(cat)
        return cat, stream
