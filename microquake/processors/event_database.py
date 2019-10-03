# This modules expect to receive a message containing the following:
# [catalog, stream, context_stream, event_id]

from loguru import logger
from microquake.clients.api_client import (post_data_from_objects)

from microquake.core.settings import settings
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "event_database"

    def initializer(self):
        self.api_base_url = settings.API_BASE_URL

    def process(self, **kwargs):
        cat = kwargs.get('cat')
        stream = kwargs.get('stream')
        variable_length = kwargs.get('variable_length')
        context = kwargs.get('context')

        logger.info('posting data to the API')

        logger.info('posting seismic data')
        result = post_data_from_objects(self.api_base_url, event_id=None,
                                        cat=cat, stream=stream,
                                        context=context,
                                        variable_length=variable_length,
                                        tolerance=None, send_to_bus=False)

        if 199 < result.status_code < 300:
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
