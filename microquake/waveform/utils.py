# -*- coding: utf-8; -*-
#
# (c) 2016 microquake development team
#
# This file is part of the microquake library
#
# microquake is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# microquake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with microquake.  If not, see <http://www.gnu.org/licenses/>.


def snr(stream, event, pre_wl=10e-3, post_wl=10e-3):
    """
    :param stream: stream associated to the event
    :type stream: microquake.core.Stream
    :param event: event
    :type event: microquake.core.event.Event
    :param pre_wl: Noise's energy is calculated from pick - pre_wl to pick  
    :type pre_wl: float
    :param post_wl: Signal's energy is calculated from pick to pick + post_wl
    :return: a pick object with an extra parameter containing the signal to noise ratio
    """

    import numpy as np

    stream = stream.detrend('linear').detrend('demean')

    for _i, pick in enumerate(event.picks):
        st = stream.select(station=pick.waveform_id.station_code)
        if not st:
            continue

        enoise = 0
        esignal = 0
        for tr in st:
            noise = tr.copy()
            signal = tr.copy()
            noise = noise.trim(starttime=pick.time - pre_wl, endtime=pick.time)
            signal = signal.trim(starttime=pick.time, endtime=pick.time + post_wl)

            enoise += np.var(noise.data)
            esignal += np.var(signal.data)

        signal_to_noise = 10 * np.log10(esignal/enoise)

    return event

def mccc(stream, event, ctime=[-0.02, 0.03], snr_th=9, freqmin=60, freqmax=1000):
    """
    :param stream: stream containing the seismograms for the event
    :type stream: micqoquake.core.Stream
    :param event: event file
    :type event: microquake.core.event.Event
    :param ctime: correlation times around the picks
    :param ctime: list
    :param freqmin: minimum frequency of bandpass filter
    :param freqmax: maximum frequency of bandpass filter
    :param snr_th: signal to noise ration threshold in dB
    :return: a pick object of corrected arrival times
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from progress.bar import Bar

    from scipy.special import comb

    dt = stream[0].stats.delta

    data_sum = np.zeros(np.floor((ctime[1] - ctime[0]) / dt))
    st = stream.copy()
    st.filter('bandpass',freqmin = freqmin, freqmax=freqmax)
    st.detrend('linear').detrend('demean')

    # separate p and s picks

    for phase in ['P', 'S']:
        p_picks = []
        indices = []
        for _i, pick in enumerate(event.picks):
            if pick.phase_hint == phase:
                p_picks.append(pick)
                indices.append(_i)

        if len(p_picks) < 2:
            continue

        dt_ij = np.zeros([len(p_picks), len(p_picks)])
        r_ij = np.zeros([len(p_picks), len(p_picks)])


        bar = Bar("MCCC", max=comb(len(pick), 2))
        for _i, pick_i in enumerate(p_picks[:-1]):
            for _j, pick_j in enumerate(p_picks[_i+1:]):
                # will need to rotate the component
                st_i = st.select(station=pick_i.waveform_id.station_code).copy()
                # st_i_p = rotate(st_i)
                st_i = st_i.trim(starttime=pick_i.time - ctime[0],
                                 endtime=pick_i.time + ctime[1]).composite()
                st_i = st_i.taper(type='cosine', max_percentage=0.05)

                st_j = st.select(station=pick_j.waveform_id.station_code).copy()
                st_j = st_j.trim(starttime=pick_j.time - ctime[0],
                                 endtime=pick_j.time + ctime[1]).composite()
                st_j = st_j.taper(type='cosine', max_percentage=0.05)

                npts = 1024
                CC = np.fft.fft(st_i[0].data, n=npts) * np.conj(np.fft.fft(st_j[0].data, n=npts))
                cc = np.fft.fftshift(np.real(np.fft.ifft(CC))) / len(st_i[0].data)
                tau_ij_max = (np.argmax(cc) - npts / 2.) * st_i[0].stats.delta
                dt_ij[_i,_j] = pick_i.time - pick_j.time - tau_ij_max
                r_ij[_i,_j] = np.max(cc) / (np.std(st_i[0].data) * np.std(st_j[0].data))
                bar.next()

        z = 1/2. * np.log((1 + r_ij) / (1 - r_ij))

        # building the matrix
        nrow = 0
        for n in range(0, len(p_picks)):
            nrow += n
        nrow += 1
        ncolumn = len(p_picks)

        A = np.zeros([nrow, ncolumn])
        A[-1,:] = 1
        B = np.zeros(nrow).T

        row_ct = 0
        for _i in range(0, len(p_picks) - 1):
            for _j in range(_i + 1, len(p_picks)):
                A[row_ct, _j] = -1
                A[row_ct, _i] = 1

                B[row_ct] = dt_ij[_i, _j]

                row_ct += 1

        A = np.matrix(A)
        B = np.matrix(B)
        t_est_mat = np.linalg.inv(A.T * A) * A.T * B.T
        t_est = np.array(t_est_mat)
        res = np.array(A * t_est_mat - B.T)

        sigma = np.zeros(len(t_est))
        try:
            for _k in range(0, len(sigma)):
                sigma[_k] = np.sqrt(1./(len(sigma) - 2) * \
                                    np.sum(res[:_k] ** 2) + np.sum([res[_k + 1:] ** 2]))
        except:
            continue

        # t_est += np.abs(np.min(t_est))
        for _k, (t, r) in enumerate(zip(t_est, res)):
            event.picks[indices[_k]].time += t[0]
            # event.picks[indices[_k]].mccc_dt_uncertainty = r[0]


    return event




