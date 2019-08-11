# This module calculates the magnitude base of a simple GMPE
import numpy as np
from microquake.core.event import Magnitude, QuantityError
from microquake.core.settings import settings

from microquake.processors.processing_unit import ProcessingUnit

class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "quick_magnitude"

    def process(
            self,
            **kwargs
    ):
        """
        Calculate a quick value for magnitude base on GMPE equation
        :param kwargs:
        :return:
        """

        stream = kwargs['stream']
        cat = kwargs['cat']

        mags = []

        # This function will need to return a tuple with station code and
        # location
        stations = stream.unique_stations()

        location = ''

        ev_loc = cat[0].preferred_origin().loc

        inventory = settings.inventory

        for station in stations:
            st_stream = stream.select(station=station,
                                      location=location).copy()

            st_loc = inventory.select(station).loc
            dist = np.linalg.norm(ev_loc - st_loc)

            motion_type = inventory.select(station).motion
            if motion_type != 'VELOCITY':
                if motion_type == 'ACCELERATION':
                    velocity = st_stream.composite().copy().integrate()[0].data
                else:
                    continue

            else:
                velocity = st_stream.composite().copy()[0].data


            # The PPV/Mag relationship is valid for mm/s, the velocity is
            # expressed in m/s in the file, we need to multiply by 1000
            ppv = np.max(velocity) * 1000 * dist

            mags.append((np.log10(ppv) - self.params.c) / self.params.a)

        mags = np.array(mags)[np.nonzero(np.isnan(mags) == False)[0]]

        self.mag = np.median(mags)
        self.mag_uncertainty = np.std(mags)
        self.station_count = len(mags)

        return (self.mag, self.mag_uncertainty)

    def output_catalog(self, catalog):

        catalog = catalog.copy()
        error = QuantityError(uncertainty=self.mag_uncertainty)
        mag = Magnitude(mag=self.mag, magnitude_type='Mw',
                        evaluation_station='automatic',
                        evaluation_status='preliminary',
                        station_count = self.station_count,
                        mag_errors=error,
                        origin_id=catalog[0].preferred_origin_id)

        catalog[0].magnitudes.append(mag)
        catalog[0].preferred_magnitude_id = mag.resource_id

        return catalog

