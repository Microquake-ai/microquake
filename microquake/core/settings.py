import os

from dynaconf import LazySettings
from loguru import logger
from microquake.core.data.inventory import Inventory


class Settings(LazySettings):
    def __init__(self):
        """
        Init function currently just initializes the object allowing
        """
        if "SPP_CONFIG" in os.environ:
            # keep thpis as legacy behavior
            config_dir = os.environ['SPP_CONFIG']
        else:
            config_dir = os.getcwd()

        dconf = {}
        dconf.setdefault('ENVVAR_PREFIX_FOR_DYNACONF', 'SPP')

        env_prefix = '{0}_ENV'.format(
            dconf['ENVVAR_PREFIX_FOR_DYNACONF']
        )  # SPP_ENV

        dconf.setdefault(
            'ENV_FOR_DYNACONF',
            os.environ.get(env_prefix, 'DEVELOPMENT').upper()
        )

        # This was an incredibly odd fix, the base settings.toml needs to be on top of the list
        # otherwise you will not be able to modify the settings downstream
        default_paths = (
            f"{os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings.toml')},"
            "settings.py,.secrets.py,"
            "settings.toml,settings.tml,.secrets.toml,.secrets.tml,"
            "settings.yaml,settings.yml,.secrets.yaml,.secrets.yml,"
            "settings.ini,settings.conf,settings.properties,"
            "connectors.toml,connectors.tml,.connectors.toml,.connectors.tml,"
            "connectors.json,"
            ".secrets.ini,.secrets.conf,.secrets.properties,"
            "settings.json,.secrets.json"
        )

        dconf['SETTINGS_FILE_FOR_DYNACONF'] = default_paths
        dconf['ROOT_PATH_FOR_DYNACONF'] = config_dir

        super().__init__(**dconf)

        self.config_dir = config_dir

        if hasattr(self, "COMMON"):
            self.common_dir = self.COMMON
        elif hasattr(self, "SPP_COMMON"):
            self.common_dir = self.SPP_COMMON

        if not self.get('common_dir', ''):
            logger.warning("Missing SPP_COMMON in env")

        self.nll_base = os.path.join(self.common_dir,
                                     self.get('nlloc').nll_base)
        self.grids = self.get('grids')

        self.sensors = self.get('sensors')

        if self.sensors.source == 'local':
            # MTH: let's read in the stationxml directly for now!
            fpath = os.path.join(self.common_dir, self.sensors.stationXML)
            self.inventory = Inventory.load_from_xml(fpath)

            # fpath = os.path.join(settings.common_dir, sensors.path)
            # self.inventory = load_inventory(fpath, format='CSV')

        elif self.sensors.get('sensors').source == 'remote':
            self.inventory = None


settings = Settings()
