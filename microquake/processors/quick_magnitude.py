# This module calculates the magnitude base of a simple GMPE
import numpy as np
from obspy.core.event import QuantityError

from microquake.core.event import Magnitude
from microquake.processors.processing_unit import ProcessingUnit
from microquake.waveform.mag_utils import calc_static_stress_drop


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
        fcs = []
        ppvs = []

        # This function will need to return a tuple with station code and
        # location
        stations = stream.unique_stations()

        location = ''

        ev_loc = cat[0].preferred_origin().loc

        inventory = self.settings.inventory

        for station in stations:
            if inventory.select(station) is None:
                continue

            st_stream = stream.select(station=station,
                                      location=location).copy()

            st_loc = inventory.select(station).loc
            dist = np.linalg.norm(ev_loc - st_loc)

            motion_type = inventory.select(station).motion

            st_stream = st_stream.detrend('demean')

            if motion_type != 'VELOCITY':
                if motion_type == 'ACCELERATION':
                    velocity = st_stream.composite().detrend(
                        'demean').copy().integrate()[0].data
                else:
                    continue

            else:
                velocity = st_stream.composite().detrend('demean').copy()[
                    0].data

            # finding the dominant frequency as an estimate of the corner
            # frequency

            v_tr = st_stream.composite()[0]

            n_max = np.argmax(np.abs(v_tr.data))

            buffer = 0.03
            t_left = v_tr.stats.starttime + n_max * v_tr.stats.delta - buffer
            t_right = v_tr.stats.starttime + n_max * v_tr.stats.delta + \
                      buffer

            v = v_tr.copy()
            v = v.trim(starttime=t_left, endtime=t_right).detrend(
                'demean').taper(max_percentage=0.1)

            v_data = v.data

            len_sign = int(2 ** np.ceil(np.log10(len(v_tr.data)) / np.log10(
                2)))
            velocity_f = np.fft.fft(v_data, len_sign)
            f = np.fft.fftfreq(len_sign, d=st_stream[0].stats.delta)
            i = np.argmax(np.abs(velocity_f[0:int(len_sign/2)]))
            fcs.append(f[i])



            # The PPV/Mag relationship is valid for mm/s, the velocity is
            # expressed in m/s in the file, we need to multiply by 1000
            ppv = np.max(np.abs(velocity)) * 1000 * dist
            ppvs.append(np.max(np.abs(velocity)))

            mags.append((np.log10(ppv) - self.params.c) / self.params.a)

        mags = np.array(mags)[np.nonzero(np.isnan(mags) == False)[0]]
        indices = np.argsort(ppvs)

        # use the station with the 10 largest PPV to measure the dominant
        # frequency
        self.corner_frequency_hz = np.median(np.array(fcs)[indices[-10:]])

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
                        station_count=self.station_count,
                        mag_errors=error,
                        origin_id=catalog[0].preferred_origin_id,
                        corner_frequency_hz=self.corner_frequency_hz)
        mag.seismic_moment = 10 ** (3 / 2 * (self.mag + 6.02))
        mag.potency_m3 = mag.seismic_moment / 29.5e9
        ssd = calc_static_stress_drop(mag.mag, mag.corner_frequency_hz)
        mag.static_stress_drop_mpa = ssd
        mag.corner_frequency_hz = self.corner_frequency_hz

        catalog[0].magnitudes.append(mag)
        catalog[0].preferred_magnitude_id = mag.resource_id

        return catalog
