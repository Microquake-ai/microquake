import numpy as np

from loguru import logger
from microquake.waveform.mag import (calc_magnitudes_from_lambda,
                                     set_new_event_mag)

from microquake.core.helpers.velocity import get_velocities
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "magnitude"

    def initializer(self):
        self.vp_grid, self.vs_grid = get_velocities()

    def process(
        self,
        **kwargs
    ):
        """
        process(catalog)

        Calculates the Magnitude in Frequency or Time domain

        - various measures
        - requires the arrivals

        Parameters
        ----------
        catalog: str

        Returns
        -------
        catalog: str

        few parameters related to the magitude
        list of magnitudes for each stations
        """
        logger.info("pipeline: magnitude")

        cat = kwargs["cat"].copy()

        density = self.params.density
        min_dist = self.params.min_dist
        use_sdr_rad = self.params.use_sdr_rad
        use_free_surface_correction = self.params.use_free_surface_correction
        make_preferred = self.params.make_preferred
        phase_list = self.params.phase_list
        use_smom = self.params.use_smom

        if not isinstance(phase_list, list):
            phase_list = [phase_list]

        if use_sdr_rad and cat.preferred_focal_mechanism() is None:
            logger.warning("use_sdr_rad=True but preferred focal mech = None"
                           "--> Setting use_sdr_rad=False")
            use_sdr_rad = False

        for i, event in enumerate(cat):

            ev_loc = event.preferred_origin().loc
            vp = self.vp_grid.interpolate(ev_loc)[0]
            vs = self.vs_grid.interpolate(ev_loc)[0]

            sdr = None

            if use_sdr_rad:
                focal_mech = event.preferred_focal_mechanism()

                if focal_mech is not None:
                    nodal_plane = focal_mech.nodal_planes.nodal_plane_1
                    strike = nodal_plane.strike
                    dip = nodal_plane.dip
                    rake = nodal_plane.rake
                    sdr = (strike, dip, rake)
                    logger.info("use_sdr_rad=True (s,d,r)=(%.1f,%.1f,%.1f)" %
                                (strike, dip, rake))

            Mws = []
            station_mags = []

            for phase in phase_list:

                Mw, sta_mags = calc_magnitudes_from_lambda(
                    [event],
                    vp=vp,
                    vs=vs,
                    density=density,
                    P_or_S=phase,
                    use_smom=use_smom,
                    use_sdr_rad=use_sdr_rad,
                    use_free_surface_correction=use_free_surface_correction,
                    sdr=sdr,
                    min_dist=min_dist)

                Mws.append(Mw)
                station_mags.extend(sta_mags)

                logger.info("Mw_%s=%.1f len(station_mags)=%d" %
                            (phase, Mws[-1], len(station_mags)))

            if self.module_type == "frequency":
                Mw = np.nanmean(Mws)
                comment = "frequency-domain"
            else:
                Mw = np.mean(Mws)
                comment = "time-domain"

            comment = f"Average of {comment} station moment magnitudes"

            if use_sdr_rad and sdr is not None:
                comment += " Use_sdr_rad: sdr=(%.1f,%.1f,%.1f)" % (sdr[0], sdr[1], sdr[2])

            if np.isnan(Mw):
                logger.warning("Mw is nan, cannot set on event")

                continue

            set_new_event_mag(event, station_mags, Mw, comment,
                              make_preferred=make_preferred)

        return cat.copy()

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        _, stream = self.app.deserialise_message(msg_in)

        return res['cat'], stream
