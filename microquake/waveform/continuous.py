# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: continuous.py
#  Purpose: module to process continuous data
#   Author: microquake development team
#    Email: dev@microquake.org
#
# Copyright (C) 2008-2012 Lion Krischer
# --------------------------------------------------------------------
"""
module to process continuous data

:copyright:
    microquake development team (dev@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from microquake.core import read
from microquake.core.stream import Stream
import numpy as np


def trigger(st, STALTA_on=1.5, STALTA_off=1.0, STA=10e-3, LTA=100e-3,
        max_trigger_length=0.1, **kwargs):
    """
    return triggers in continuous recordings
    """
    from microquake.signal.trigger import recursive_sta_lta
    from microquake.signal.trigger import trigger_onset
    from microquake.core.stream import composite_traces
    import matplotlib.pyplot as plt

    cft = np.zeros(len(st.traces[0]))
    trigs = []
    st_comp = composite_traces(st)
    trs = []
    for tr in st_comp:
        nsta = int(STA * tr.stats.sampling_rate)
        nlta = int(LTA * tr.stats.sampling_rate)
        cft = recursive_sta_lta(np.array(tr.data), nsta, nlta)
        trigs_tmp = trigger_onset(cft, STALTA_on, STALTA_off,
                    max_len=max_trigger_length * tr.stats.sampling_rate,
                    max_len_delete=True)

        for trig in trigs_tmp:
            for trg in trig:
                trs.append(tr)
            trigs.append(trig)
    trigs = np.array(trigs).ravel()
    # cft /= len(st)
    # trigs = trigger_onset(cft, STALTA_on, STALTA_off,
        # max_len=max_trigger_length * tr.stats.sampling_rate,
        # max_len_delete=True)
    if np.any(trigs):
        trigs = np.array([(trs[k].stats.station, trs[k].stats.starttime + trg /
            trs[k].stats.sampling_rate) for k, trg in enumerate(trigs)])

    tmp = {}
    tmp['time'] = trigs[:,1]
    tmp['station'] = trigs[:,0]
    trigs = tmp

    return trigs


def associate(trigs, association_time=0.1, hold_off=0.2):
    """
    Associates trigger times
    """
    from scipy.cluster.vq import kmeans

    i = np.argsort(trigs['time'])
    trgtimes = np.array(trigs['time'])[i]
    stations = np.array(trigs['station'])[i]

    tt = (trgtimes - trgtimes[0]).astype(float)
    k = len(np.nonzero(np.diff(tt) > association_time)[0]) + 1
    centtimes = np.array(trgtimes[0]) + np.sort(kmeans(tt, k)[0])

    associate_time = []
    trigs_times = []
    trigs_stations = []

    for ct in centtimes:
        indices = np.nonzero(np.abs((trgtimes - ct).astype(float)) <=
                association_time)[0]
        if len(indices) >= 5:
            associate_time.append(ct)
            trigs_times.append(trgtimes[indices])
            trigs_stations.append(stations[indices])

    if len(associate_time) > 1:
        diff = np.diff(associate_time)
        index = np.nonzero(diff < hold_off)[0]
        associate_time = np.delete(associate_time, index + 1)

    return associate_time, trigs_times, trigs_stations


def event_detection(ifile, site, buf, association_time, min_n_trigs,
                 hold_off_time):
    """
        Performing event detection and association
    """
    try:

        client = MongoClient()
        nrs_db = client['nrs']
        seismogram_col = nrs_db['seismograms']
        processed_col = nrs_db['processed_files']
        d = Default(omit_nulls=False)

        cursor = processed_col.find({})
        processed_files = []
        processed_files = [entry['name'] for entry in cursor]

        if ifile in processed_files:
            return True

        st = load_data(ifile, stations, buf).copy()

        stas = np.unique([tr.stats.station for tr in st])
        trigs = {}
        trigs['station'] = []
        trigs['time'] = []
        for sta in stas:
            sttmp = st.select(station=sta).filter('highpass', freq=100)
            trgs = trigger(sttmp, buf)
            for trg in trgs:
                trigs['station'].append(sta)
                trigs['time'].append(trg)

        trigs['station'] = np.array(trigs['station'])
        trigs['time'] = np.array(trigs['time'])

        if len(np.unique(trigs['station'])) >= 5:
            associate_times, trig_times, trig_stations = associate(trigs,
                                                            association_time,
                                                            min_n_trigs,
                                                            hold_off_time)

            for k, t in enumerate(associate_times):
                sttrim = st.copy()
                sttrim.trim(starttime=t - buf, endtime=t + buf)

                sttrim.trigger_station = trig_stations[k]
                sttrim.trigger_time = trig_times[k]
                stdict = stream2dict(sttrim)
                stdict['isvalid'] = clean(SeisData(sttrim))
                stdict['trigger_station'] = list(trig_stations[k])
                stdict['trigger_times'] = list(trig_times[k])
                stdict['trigger_time'] = t
                s = json.dumps(stdict, default=d)
                tmp = loads(s)
                # datetimes are saved as string in the database

                seismogram_col.insert_one(tmp)

        processed_col.insert_one({"name": ifile})

        gc.collect()
        client.close()
        return True
    except:
        return True

