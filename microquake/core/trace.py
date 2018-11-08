# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
#  Purpose: Expansion of the obspy.core.trace module
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.trace module

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import obspy.core.trace as obstrace
from obspy.core.trace import Stats
from microquake.core.util import tools
from microquake.core.event import Pick, WaveformStreamID

import numpy as np


class Trace(obstrace.Trace):
    def __init__(self, trace=None, **kwargs):
        super(Trace, self).__init__(**kwargs)
        if trace:
            self.stats = trace.stats
            self.data = trace.data
    
    @property
    def sr(self):
        return self.stats.sampling_rate
    
    def ppv(self):
        return np.max(np.abs(self.data))

    def time_to_index(self, time):
        return int((time - self.stats.starttime) * self.sr)

    def index_to_time(self, index):
        return self.stats.starttime + (index / self.sr)

    def times(self):
        sr = self.stats.sampling_rate
        return np.linspace(0, len(self.data) / sr, len(self.data))

    def plot(self, **kwargs):
        from microquake.imaging.waveform import WaveformPlotting
        waveform = WaveformPlotting(stream=self, **kwargs)
        return waveform.plotWaveform()

    def make_pick(self, pick_time, wlen_search,
                 stepsize, snr_wlens, phase_hint=None):

        ipick = self.time_to_index(pick_time)
        sr = self.stats.sampling_rate
        stepsize_samp = int(stepsize * sr)
        snr_wlens_samp = (snr_wlens * sr).astype(int)
        wlen_search_samp = int(wlen_search * sr)

        newpick, snr = tools.repick_using_snr(self.data, ipick, wlen_search_samp,
                             stepsize_samp, snr_wlens_samp)

        waveform_id = WaveformStreamID(channel_code=self.stats.channel, station_code=self.stats.station)

        pick = Pick(time=self.index_to_time(newpick), waveform_id=waveform_id, phase_hint=phase_hint, evaluation_mode='automatic', evaluation_status='preliminary', method='snr', snr=snr)

        return pick
