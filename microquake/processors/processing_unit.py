from abc import ABC, abstractmethod

from loguru import logger

from microquake.core.settings import settings


class ProcessingUnit(ABC):
    def __init__(self, input=None, output=None, app=None, module_type=None):
        self.__input = input
        self.__output = output
        self.app = app
        self.debug_level = settings.DEBUG_LEVEL
        self.debug_file_dir = settings.DEBUG_FILE_DIR
        self.settings = settings
        self.params = settings.get(self.module_name)

        """
        override parameters by module_type subparameters
        modulename.moduletype
        """
        self.set_module_type(module_type)

        logger.info("pipeline unit: {}", self.module_name)

        super(ProcessingUnit, self).__init__()
        self.initializer()

    @abstractmethod
    def module_name(self):
        pass

    @property
    def input(self):
        return self.__input

    @property
    def output(self):
        return self.__output

    def initializer(self):
        """ initialize processing unit """

    def legacy_pipeline_handler(self, msg_in, res):
        cat, stream = self.app.deserialise_message(msg_in)

        return cat, stream

    def output_catalog(self, catalog):
        return self.result['cat']

    def set_module_type(self, module_type):
        self.module_type = module_type
        extra_params = settings.get(f"{self.module_name}.{module_type}")

        if extra_params:
            self.params.update(extra_params)
