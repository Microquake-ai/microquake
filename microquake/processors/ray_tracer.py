# This module calculates the magnitude base of a simple GMPE

from microquake.processors.processing_unit import ProcessingUnit
from loguru import logger

from microquake.core.simul.eik import ray_tracer
from microquake.core.helpers.grid import get_ray, get_grid_point


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "ray_tracer"

    def process(
            self,
            **kwargs
    ):

        cat = kwargs['cat']

        p_ori = cat[0].preferred_origin()
        inventory = self.settings.inventory

        if p_ori is None:
            logger.warning('No preferred origin in the current event')
            return

        ev_loc = p_ori.loc

        self.rays = []
        for station in inventory.stations():
            st_loc = station.loc
            station_code = station.code
            logger.info(f'calculating rays for station {station_code}')
            for phase in ['P', 'S']:
                try:
                    ray = get_ray(station_code, phase, ev_loc)
                    ray.station_code = station_code
                    ray.phase = phase
                    ray.arrival_id = p_ori.get_arrival_id(phase, station_code)
                    ray.travel_time = get_grid_point(station_code, phase,
                                                     ev_loc, grid_type='time')
                    ray.azimuth = get_grid_point(station_code, phase,
                                                 ev_loc, grid_type='azimuth')
                    ray.takeoff_angle = get_grid_point(station_code, phase,
                                                       ev_loc,
                                                       grid_type='take_off')
                    self.rays.append(ray)
                except FileNotFoundError:
                    logger.warning(f'travel time grid for station '
                                   f'{station_code} '
                                   f'and phase {phase} was not found. '
                                   f'Skipping...')

        return self.rays

    def output_catalog(self, catalog):
        for ray in self.rays:
            catalog[0].preferred_origin().append_ray(ray)

        return catalog
