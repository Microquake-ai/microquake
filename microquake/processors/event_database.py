# This modules expect to receive a message containing the following:
# [catalog, stream, context_stream, event_id]

from loguru import logger
from spp.utils.seismic_client import (post_data_from_objects)

from microquake.core.settings import settings
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "event_database"

    def initializer(self):
        self.api_base_url = settings.API_BASE_URL

    def process(
        self,
        **kwargs
    ):
        cat = None
        stream = None
        variable_length = None
        context = None
        method = None # POST or PUT

        if 'cat' in kwargs.keys():
            cat = kwargs['cat']
        if 'stream' in kwargs.keys():
            stream = kwargs['stream']
        if 'variable_length' in kwargs.keys():
            variable_length = kwargs['variable_length']
        if 'context' in kwargs.keys():
            context = kwargs['context']


        logger.info('posting data to the API')

        logger.info('posting seismic data')
        result = post_data_from_objects(self.api_base_url, event_id=None,
                                        event=cat,
                                        stream=stream,
                                        context_stream=context,
                                        variable_length_stream=variable_length,
                                        tolerance=None,
                                        send_to_bus=False)


        if result.status_code == 200:
            logger.info('successfully posting data to the API')
        else:
            logger.error('Error in sending data to the API. Returned with '
                         'error code %s' % result)

        return result

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        return res['cat'], res['stream']
