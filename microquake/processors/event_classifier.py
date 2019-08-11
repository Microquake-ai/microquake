from spp.classifier.seismic_classifier import SeismicClassifierModel
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    """
        Class wrapper around SeismicClassifierModel, load inputs from kwargs.
    """
    @property
    def module_name(self):
        return "event_classifier"

    def initializer(self):
        self.seismic_model = SeismicClassifierModel()
        self.seismic_model.create_model()

    def process(self, **kwargs):
        """
            Process event and returns its classification.
        """
        stream = kwargs["stream"]
        height = kwargs["height"]
        context_trace = kwargs["context"]
        magnitude = kwargs["magnitude"]
        self.response = self.seismic_model.predict(stream, context_trace,
                                                   height, magnitude)
        return self.response

    def legacy_pipeline_handler(self, msg_in, res):
        """
            legacy pipeline handler
        """
        cat, stream = self.app.deserialise_message(msg_in)
        cat = self.output_catalog(cat)
        return cat, stream
