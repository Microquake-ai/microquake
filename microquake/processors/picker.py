from time import time

import numpy as np
from obspy import UTCDateTime
from obspy.core.event import CreationInfo

from loguru import logger
from microquake.core.event import Origin
from microquake.core.helpers.grid import (create_arrivals_from_picks,
                                          estimate_origin_time,
                                          synthetic_arrival_times)
from microquake.processors.processing_unit import ProcessingUnit
from microquake.waveform.pick import snr_picker


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "picker"

    def initializer(self):
        self.freq_min = self.params.waveform_filter.frequency_min
        self.freq_max = self.params.waveform_filter.frequency_max
        self.residual_tolerance = self.params.residual_tolerance
        self.p_wave_window_start = self.params.p_wave.search_window.start
        self.p_wave_window_end = self.params.p_wave.search_window.end
        self.p_wave_window_resolution = self.params.p_wave.search_window.resolution
        self.p_wave_noise = self.params.p_wave.snr_window.noise
        self.p_wave_signal = self.params.p_wave.snr_window.signal
        self.s_wave_window_start = self.params.s_wave.search_window.start
        self.s_wave_window_end = self.params.s_wave.search_window.end
        self.s_wave_window_resolution = self.params.s_wave.search_window.resolution
        self.s_wave_noise = self.params.s_wave.snr_window.noise
        self.s_wave_signal = self.params.s_wave.snr_window.signal
        self.snr_threshold = self.params.snr_threshold

    def process(
        self,
        **kwargs
    ):
        """
        Predict picks for event
        takes waveform and location

        inputs:
           - Location as list [x, y, z]
           - Origin time UTC as UTCDateTime
           - fixed_length stream as microquake.core.stream.Stream object

        list of picks
        list of phase and time
        """
        # cat = kwargs["cat"]
        stream = kwargs['stream'].copy()
        phase_filter = None

        if "location" and "event_time_utc" in kwargs:
            o_loc = kwargs['location']  # a list containing the location
            ot_utc = kwargs['event_time_utc']
        elif "cat" in kwargs:
            cat = kwargs["cat"]
            if cat[0].preferred_origin_id is None:
                o_loc = cat[0].origins[-1].loc
                ot_utc = estimate_origin_time(stream, o_loc)
            else:
                o_loc = cat[0].preferred_origin().loc
                ot_utc = cat[0].preferred_origin().time
        else:
            raise Exception("Missing parameters to run this module.")

        if 'phase_filter' in kwargs:
            phase_filter = kwargs['phase_filter']

        logger.info('cleaning the input stream')
        st = stream.detrend("demean")
        logger.info('done cleaning the input stream. %d of %d stations kept.' %
                    (len(st.unique_stations()), len(stream.unique_stations())))

        st = st.taper(max_percentage=0.1, max_length=0.01)
        st = st.filter("bandpass", freqmin=self.freq_min,
                       freqmax=self.freq_max)

        logger.info("predicting picks for origin time {}", ot_utc)
        t2 = time()
        picks = synthetic_arrival_times(o_loc, ot_utc)
        t3 = time()
        logger.info("done predicting picks in %0.3f seconds" % (t3 - t2))

        snrs_p = []
        snrs_s = []
        p_snr_picks = []
        s_snr_picks = []

        if phase_filter is None:
            phase_filter = 'PS'

        if 'P' in phase_filter.upper():
            logger.info("picking P-waves")
            t4 = time()
            search_window = np.arange(
                self.p_wave_window_start,
                self.p_wave_window_end,
                self.p_wave_window_resolution,
            )

            snr_window = (
                self.p_wave_signal,
                self.p_wave_noise,
            )

            st_c = st.copy().composite()
            snrs_p, p_snr_picks = snr_picker(
                st_c, picks, snr_dt=search_window, snr_window=snr_window,
                filter="P"
            )

            t5 = time()
            logger.info("done picking P-wave in %0.3f seconds" % (t5 - t4))

        if 'S' in phase_filter.upper():
            logger.info("picking S-waves")
            t6 = time()

            search_window = np.arange(
                self.s_wave_window_start,
                self.s_wave_window_end,
                self.s_wave_window_resolution,
            )

            snr_window = (
                self.s_wave_signal,
                self.s_wave_noise,
            )

            snrs_s, s_snr_picks = snr_picker(
                st_c, picks, snr_dt=search_window, snr_window=snr_window,
                filter="S"
            )
            t7 = time()

            logger.info("done picking S-wave in %0.3f seconds" % (t7 - t6))

        snr_picks = p_snr_picks + s_snr_picks

        if len(snr_picks) == 0:
            return False

        snrs = snrs_p + snrs_s

        snr_picks_filtered = [
            snr_pick
            for (snr_pick, snr) in zip(snr_picks, snrs)
            if snr > self.snr_threshold
        ]

        # no picks for sensors in black list
        black_list = self.settings.get('sensors').black_list
        snr_picks_filtered = [
            snr_pick
            for snr_pick in snr_picks_filtered
            if snr_pick.waveform_id.station_code not in black_list
        ]

        t0 = time()
        residuals = []

        for snr_pk in snr_picks_filtered:
            for pk in picks:
                if (pk.phase_hint == snr_pk.phase_hint) and (
                    pk.waveform_id.station_code ==
                    snr_pk.waveform_id.station_code
                ):
                    residuals.append(pk.time - snr_pk.time)

        residuals = np.array(residuals)
        residuals -= np.mean(residuals)

        indices = np.nonzero(np.abs(residuals) < self.residual_tolerance)[0]
        snr_picks_filtered = [snr_picks_filtered[i] for i in indices]

        t1 = time()

        logger.info("creating arrivals")
        t8 = time()
        arrivals = create_arrivals_from_picks(snr_picks_filtered, o_loc,
                                              ot_utc)

        t9 = time()
        logger.info("done creating arrivals in %0.3f seconds" % (t9 - t8))

        logger.info("creating new event or appending to existing event")
        t10 = time()

        t11 = time()

        logger.info("Origin time: {}", ot_utc)
        logger.info("Total number of picks: %d" % len(arrivals))

        logger.info(
            "done creating new event or appending to existing event "
            "in %0.3f seconds" % (t11 - t10)
        )

        origin = Origin()
        origin.time = ot_utc
        origin.x = o_loc[0]
        origin.y = o_loc[1]
        origin.z = o_loc[2]
        origin.arrivals = arrivals
        origin.evaluation_mode = "automatic"
        origin.evaluation_status = "preliminary"
        origin.creation_info = CreationInfo(creation_time=UTCDateTime.now())
        origin.method_id = "PICKER_FOR_HOLDING_ARRIVALS"

        self.response = {'picks': snr_picks_filtered,
                         'origins': [origin],
                         'preferred_origin_id': origin.resource_id.id}

        return self.response

    def output_catalog(self, catalog):
        catalog = catalog.copy()
        picks = self.response['picks']
        origins = self.response['origins']
        preferred_origin_id = self.response['preferred_origin_id']

        catalog[0].picks += picks
        catalog[0].origins += origins
        catalog[0].preferred_origin_id = preferred_origin_id

        return catalog

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        cat, stream = self.app.deserialise_message(msg_in)

        cat = self.output_catalog(cat)

        return cat, stream
