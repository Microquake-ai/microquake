"""
Predict hypocenter location
"""

from time import time

from loguru import logger
from microquake.core.nlloc import NLL, calculate_uncertainty

from microquake.core.helpers.grid import fix_arr_takeoff_and_azimuth
from microquake.core.helpers.nlloc import nll_sensors, nll_velgrids
from microquake.core.helpers.velocity import get_velocities
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "nlloc"

    def initializer(self):
        self.vp_grid, self.vs_grid = get_velocities()

        self.project_code = self.settings.PROJECT_CODE
        self.base_folder = self.settings.nll_base
        gridpar = nll_velgrids()
        sensors = nll_sensors()

        logger.info("preparing NonLinLoc")
        self.nll = NLL(
            self.project_code,
            base_folder=self.base_folder,
            gridpar=gridpar,
            sensors=sensors,
            params=self.params,
        )
        logger.info("done preparing NonLinLoc")

    def process(
        self,
        **kwargs
    ):
        """
        requires an event
        input: catalog

        montecarlo sampling, change the input get many locations
        many locations will form a point cloud

        change the x, y,z many times

        returns: x,y,z,time, uncertainty
        uncertainty measure the effect of errors of the measurements on the results
        measurement picks, results is the location

        returns point cloud
        """
        logger.info("pipeline: nlloc")

        cat = kwargs["cat"]

        logger.info("running NonLinLoc")
        t0 = time()
        cat_out = self.nll.run_event(cat[0].copy())
        t1 = time()
        logger.info("done running NonLinLoc in %0.3f seconds" % (t1 - t0))

        if cat_out[0].preferred_origin():
            logger.info("preferred_origin exists from nlloc:")
        else:
            logger.info("No preferred_origin found")

        logger.info("calculating Uncertainty")
        t2 = time()
        picking_error = self.params.picking_error
        origin_uncertainty = calculate_uncertainty(
            cat_out[0],
            self.base_folder,
            self.project_code,
            perturbation=5,
            pick_uncertainty=picking_error,
        )

        if cat_out[0].preferred_origin():
            cat_out[0].preferred_origin().origin_uncertainty = \
                origin_uncertainty
            t3 = time()
            logger.info("done calculating uncertainty in %0.3f seconds"
                        % (t3 - t2))

        fix_arr_takeoff_and_azimuth(cat_out, self.vp_grid, self.vs_grid)

        if cat_out[0].preferred_origin():
            origin = cat_out[0].preferred_origin()

            for arr in origin.arrivals:
                arr.hypo_dist_in_m = arr.distance

        self.result = {'cat': cat_out}
        return self.result

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        _, stream = self.app.deserialise_message(msg_in)

        return res['cat'], stream
