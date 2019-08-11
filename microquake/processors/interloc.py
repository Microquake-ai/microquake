import os
from datetime import datetime
from time import time

import numpy as np
from obspy.core import UTCDateTime
from obspy.core.util.attribdict import AttribDict
from xseis2 import xspy

from loguru import logger
from microquake.core.event import Origin
from microquake.core.util import tools

from microquake.core.helpers.hdf5 import get_ttable_h5
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "interloc"

    def initializer(self):
        self.htt = get_ttable_h5()

    def process(
        self,
        **kwargs
    ):
        logger.info("pipeline: interloc")

        # TODO: copy not necessary testapplication is broken
        stream = kwargs["stream"].copy()

        nthreads = self.params.nthreads
        fixed_wlen_sec = self.params.fixed_wlen_sec
        samplerate_decimated = self.params.samplerate_decimated
        pair_dist_min = self.params.pair_dist_min
        pair_dist_max = self.params.pair_dist_max
        cc_smooth_length_sec = self.params.cc_smooth_length_sec

        whiten_corner_freqs = np.array(self.params.whiten_corner_freqs,
                                       dtype=np.float32)

        stalocs = self.htt.locations
        ttable = (self.htt.hf["ttp"][:] * samplerate_decimated).astype(np.uint16)
        ngrid = ttable.shape[1]
        ttable_row_ptrs = np.array(
            [row.__array_interface__["data"][0] for row in ttable])

        logger.info("preparing data for Interloc")
        t4 = time()

        # remove channels which do not have matching ttable entries
        # This should be handled upstream

        for trace in stream:
            if trace.stats.station not in self.htt.stations:
                logger.info('removing trace for station %s' % trace.stats.station)
                stream.remove(trace)
            elif np.max(trace.data) == 0:
                logger.info('removing trace for station %s' % trace.stats.station)
                stream.remove(trace)
            elif trace.stats.station in self.settings.sensors.black_list:
                logger.info('removing trace for station %s' % trace.stats.station)
                stream.remove(trace)

        data, samplerate, t0 = stream.as_array(fixed_wlen_sec)
        data = np.nan_to_num(data)
        decimate_factor = int(samplerate / samplerate_decimated)
        data = tools.decimate(data, samplerate, decimate_factor)
        channel_map = stream.channel_map().astype(np.uint16)

        ikeep = self.htt.index_sta(stream.unique_stations())
        debug_file = os.path.join(self.debug_file_dir, "iloc_" + str(t0) + ".npz")
        t5 = time()
        logger.info(
            "done preparing data for Interloc in %0.3f seconds" % (t5 - t4))

        logger.info("Locating event with Interloc")
        t6 = time()
        logger.info(
            "samplerate_decimated {}, ngrid {}, nthreads {}, debug {}, debug_file {}",
            samplerate_decimated, ngrid, nthreads, self.debug_level, debug_file)

        out = xspy.pySearchOnePhase(
            data,
            samplerate_decimated,
            channel_map,
            stalocs[ikeep],
            ttable_row_ptrs[ikeep],
            ngrid,
            whiten_corner_freqs,
            pair_dist_min,
            pair_dist_max,
            cc_smooth_length_sec,
            nthreads,
            self.debug_level,
            debug_file
        )

        vmax, imax, iot = out
        normed_vmax = vmax * fixed_wlen_sec
        lmax = self.htt.icol_to_xyz(imax)
        t7 = time()
        logger.info("Done locating event with Interloc in %0.3f seconds" % (t7 - t6))

        ot_epoch = tools.datetime_to_epoch_sec(
            (t0 + iot / samplerate_decimated).datetime)

        method = "%s" % ("INTERLOC",)

        logger.info("power: %.3f, ix_grid: %d, ix_ot: %d" % (vmax, imax, iot))
        logger.info("utm_loc: {}", lmax)
        logger.info("=======================================\n")
        logger.info("VMAX over threshold (%.3f)" % (vmax))

        self.response = {'x': lmax[0],
                         'y': lmax[1],
                         'z': lmax[2],
                         'vmax': vmax,
                         'normed_vmax': normed_vmax,
                         'event_time': ot_epoch,
                         'method': method}

        return self.response

    def output_catalog(self, catalog):
        catalog = catalog.copy()

        x = self.response['x']
        y = self.response['y']
        z = self.response['z']
        vmax = self.response['vmax']
        normed_vmax = self.response['normed_vmax']
        method = self.response['method']
        event_time = UTCDateTime(datetime.fromtimestamp(self.response['event_time']))

        catalog[0].origins.append(
            Origin(x=x, y=y, z=z, time=event_time,
                   method_id=method, evalution_status="preliminary",
                   evaluation_mode="automatic")
        )
        catalog[0].preferred_origin_id = catalog[0].origins[-1].resource_id.id

        catalog[0].preferred_origin().extra.interloc_vmax \
            = AttribDict({'value': vmax, 'namespace': 'MICROQUAKE'})

        catalog[0].preferred_origin().extra.interloc_normed_vmax \
            = AttribDict({'value': normed_vmax, 'namespace': 'MICROQUAKE'})

        return catalog

    def legacy_pipeline_handler(
        self,
        msg_in,
        res
    ):
        cat, stream = self.app.deserialise_message(msg_in)

        cat = self.output_catalog(cat)

        return cat, stream
