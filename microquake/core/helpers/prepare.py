from loguru import logger

from microquake.core.helpers.hdf5 import write_ttable_h5
from microquake.core.helpers.nlloc import nll_velgrids, nll_sensors
from microquake.core.nlloc import NLL
from microquake.core.settings import settings


def prepare():
    """
    Prepare project and run NonLinLoc
    """
    project_code = settings.PROJECT_CODE
    base_folder = settings.nll_base
    gridpar = nll_velgrids()
    sensors = nll_sensors()
    params = settings.get('nlloc')

    nll = NLL(project_code, base_folder=base_folder, gridpar=gridpar,
              sensors=sensors, params=params)

    # creating NLL base project including travel time grids
    logger.info('Preparing NonLinLoc')
    nll.prepare()
    logger.info('Done preparing NonLinLoc')

    # creating H5 grid from NLL grids
    logger.info('Writing h5 travel time table')
    write_ttable_h5()
