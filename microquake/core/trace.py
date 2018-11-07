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
import numpy as np


class Trace(obstrace.Trace):
    def __init__(self, trace=None, **kwargs):
        super(Trace, self).__init__(**kwargs)
        if trace:
            self.stats = trace.stats
            self.data = trace.data
    
    def ppv(self):
        """
        Calculate the PPV for a given trace
        :return: PPV
        """
        return np.max(np.abs(self.data))

    def snr_repick(self, pick_time, wlen_search, stepsize, snr_wlens):

        sr = self.stats.sampling_rate
        ipick = int(pick_time * sr)
        stepsize_samp = int(stepsize * sr)
        snr_wlens_samp = (snr_wlens * sr).astype(int)
        wlen_search_samp = int(wlen_search * sr)

        newpick, snr = tools.repick_using_snr(self.data, ipick, wlen_search_samp,
                             stepsize_samp, snr_wlens_samp)
        return newpick / sr, snr

    def times(self):
        sr = self.stats.sampling_rate
        return np.linspace(0, len(self.data) / sr, len(self.data))

    def plot(self, **kwargs):
        from microquake.imaging.waveform import WaveformPlotting
        waveform = WaveformPlotting(stream=self, **kwargs)
        return waveform.plotWaveform()
