import numpy as np
from scipy import optimize
from scipy.fftpack import fft, fftfreq, rfft, rfftfreq
#from scipy.optimize import fmin, fmin_powell, curve_fit

from spp.utils.application import Application
from microquake.core.util.tools import copy_picks_to_dict
import matplotlib.pyplot as plt

"""
    mag_utils - a collection of routines to assist in the moment magnitude calculation
"""

def parsevals(data, dt, nfft):
    tsum = np.sum(np.square(data))*dt
    print("Parseval's: [time] ndata=%d dt=%f sum=%f" % (data.size, dt, tsum))
    exit()

def check2():
    # Specify the parameters of a signal with a sampling frequency of 1 kHz and a signal duration of 1.5 seconds.

    Fs = 1000.           # Sampling frequency
    T = 1/Fs             # Sampling period
    L = 1500            # Length of signal
    t =[]
    s = []
    for j in range(L):
        t.append(float(j)*T)        # Time vector
        s.append(1.0*np.sin(2*np.pi*60*float(j)*T))
    #Form a signal containing a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz sinusoid of amplitude 1.
    #print(t)

    #Y = fft(s)
    #print(Y.dtype)
    #print(Y.size)
    #exit()

    #S = 0.7*np.sin(2*np.pi*50*t)
    #S = 0.7*np.sin(2*np.pi*50*t) + sin(2*pi*120*t);

    #for i,n in enumerate([256, 512, 1024, 2048, 4096, 8192, 16384]):
    for i,n in enumerate([1024, L, 2048, 4096, 8192, 16384]):
        df = 1./(float(n)*T)
        shat,freqs = unpack_rfft(rfft(s, n=n), df)
        #cmod = 2. * np.abs(shat/float(n))
        cmod = 2. * np.abs(shat)
        cmod /= float(L)
        #cmod = 2. * np.abs(shat*T)
        peak_f = freqs[np.argmax(cmod)]
        print("n=%d (cmod.size=%d) f:%.2f mean_shat=%f max_shat=%f" % (n, cmod.size, peak_f, np.mean(cmod), np.amax(cmod)))



def check(tr=None):
    '''
    It looks like the peak spec amp doesn't change much for Nfft >= 1024 (for rfft).
    And the only scaling needed on the forward FFT is dt.

    If you're going to prove Parseval's, then you need to scale rfft
    by sqrt(2/N) to get correct match to power in t-domain
    '''
    dt = 0.1
    s = [1/dt, 0, 0, 0, 0, 0, 0, 0]
    #n=128

    #df = 1./(float(n)*dt)
    #shat,freqs = unpack_rfft(rfft(s, n=n), df)
    #shat = fft(s, n=64)

    T = 1.
    s = []
    for j in range(1024):
        t = j*dt
        y = np.cos( 2*np.pi*t/T )
        s.append(y)

    if tr is not None:
        plot_signal(tr)
        dt = tr.stats.delta
        s  = tr.data

    #for i,n in enumerate([512, 1024]):
    for i,n in enumerate([256, 512, 1024, 2048, 4096, 8192, 16384]):
        df = 1./(float(n)*dt)
        shat,freqs = unpack_rfft(rfft(s, n=n), df)

        cmod = np.abs(shat)
        print("nfft=%d df=%.3f mean_shat=%f max_shat=%f" % (n, df, np.mean(cmod), np.amax(cmod)))

        if tr is None:
            continue

        mask = [all(f) for f in zip(freqs >= 100., freqs <= 200.)]
        new = shat[mask]

        max_amp = np.amax(np.abs(shat))
        print("N=%d-fft of real array contans [n=%d] pnts.  maxamp=[%12.10g]" % (n, shat.size,max_amp))
        mean_amp = np.mean(np.abs(new))
        print("new has %d pnts and mean=%f" % (new.size, mean_amp))

        plot_spec(freqs, shat)
    exit()
    #print(shat)

def main():
    '''
    for v in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        print(v, 2.*np.log10(v))
    exit()
    '''
    check2()

def peak_freq(spec_array, freqs, fmin=0., fmax=None):
    """ Return peak_f of spec_array within bounds: fmin <= freqs <= fmax
    """

    if fmax is None:
        fmax = freqs[-1]

    mask = [all(f) for f in zip(freqs >= fmin, freqs <= fmax)]

    return freqs[mask][np.argmax(spec_array[mask], axis=0)]


def stack_spectra(sta_dict):
    """ stack the pre-calculated spectra (fft) of each channel

        :param sta_dict: dictionary of pre-calculated P & S channel spectra at each station
        :return: normalized_spectrum_stack, freqs
        :rtype: (np.array, np.array)
    """

    stack = 0.
    n=0.
    for sta_code, sta in sta_dict.items():
        #print("  sta:%3s" % sta_code)
        for cha_code, cha in sta['chan_spec'].items():
            signal_fft, freqs, noise_fft = (cha['signal_fft'], cha['freqs'], cha['noise_fft'])

        # Normalize each fft individually and add to stack

            stack += np.abs(signal_fft)/np.amax(np.abs(signal_fft))

            #print("stack[100]=%12.10g signal_fft[100]=%12.10g max=%12.10g" % \
                  #(stack[100], np.abs(signal_fft[100]), np.amax(np.abs(signal_fft))))
            n += 1

    print("stack_spectra: stacked n=%d spectra" % n)
    # Return normalized stack, freq array
    return stack/np.amax(stack), freqs


def inv_station_list_to_dict(station_list):
    """ Convert station list (= list of obspy stations) to dict

        :param station_list: list of
        :return: dict
        :rtype: something
    """

    sta_meta_dict = {}
    for sta in station_list:
        chans_dict = {}
        for cha in sta:
            chans_dict[cha.code] = cha

        sta_meta_dict[sta.code] = chans_dict

    return sta_meta_dict


def get_spectra(st, event, stations, calc_displacement=False, S_win_len=.1, P_or_S='P'):
    """ Calculate the fft at each channel in the stream that has an arrival in event.arrivals

        :param st: microquake.core.stream.Stream
        :type  st: microquake.core.stream.Stream
        :param event: microquake.core.event.Event
        :type  event: microquake.core.event.Event
        :param stations: list of station metadata
        :type  stations: obspy.core.inventory.network.Network
        :param calc_displacement: If true calculate displacement spectra (else velocity)
        :type  calc_displacement: bool

        :return: sta_dict: dict of station P & S spectra for each channel
        :rtype: dict
    """

    # TODO Right now S win len = .1, may want to make this a function of distance and/or handle
    #     if win len exceeds trace len, etc

    fname = 'get_spectra'

    sta_meta_dict = inv_station_list_to_dict(stations)

    origin = event.preferred_origin() if event.preferred_origin() else event.origins[0]

    pick_dict = copy_picks_to_dict(event.picks)

# MTH: This is probably an expensive operation - would be better to create a single, travel-time server!
    app = Application()
    synthetic_picks = app.synthetic_arrival_times(origin.loc, origin.time)
    synthetic_dict = copy_picks_to_dict(synthetic_picks)

    if P_or_S == 'P':
        arrivals = [ arr for arr in origin.arrivals if \
                     arr.pick_id.get_referred_object().phase_hint == 'P' ]
    else:
        arrivals = [ arr for arr in origin.arrivals if \
                     arr.pick_id.get_referred_object().phase_hint == 'S' ]

    #print("%s: Calc [%s] spectra for [n=%d] arrivals" % (fname, P_or_S, len(arrivals)))

    arr_stations = set()
    for arr in arrivals:
        arr_stations.add(arr.pick_id.get_referred_object().waveform_id.station_code)

    st_stations = set()
    for tr in st:
        st_stations.add(tr.stats.station)

    stations_without_trace = arr_stations - st_stations

    # Only keep stations that have at least one arrival and at least one trace
    sta_codes = arr_stations.intersection(st_stations)

    # Turn into a list sorted numerically by sta code (but keep sta code = str)
    sta_codes = sorted([sta for sta in sta_codes], key=lambda x: int(x))

    # For now, let's be explicit about our assumptions:
    # For each trace, we'll use P-.01 to S-.01 to calculate the P spectrum, so
    #   let's use the max expected S-P time (at the farthest station) to fix NFFT
    max_S_P_time = 0.25
    dt   = st[0].stats.delta # For now, let's assume ALL traces are sampled at this rate
    npts = int(max_S_P_time/dt)
    nfft = npow2(npts) * 2 # Fix nfft for all calcs
    df   = 1/(float(nfft)*dt)
    print("%s: Set nfft=%d --> df=%.3f" % (fname, nfft, df))


# 1. Create a dict of keys=sta_code for all 'P' arrivals with the necessary pick/sta info inside
    sta_dict = {}
    for sta_code in sta_codes:

        d = {}

        if 'P' in pick_dict[sta_code]:
            d['ptime']   = pick_dict[sta_code]['P'].time
        else:
            d['ptime']   = synthetic_dict[sta_code]['P'].time

        if 'S' in pick_dict[sta_code]:
            d['stime']   = pick_dict[sta_code]['S'].time
        else:
            d['stime']   = synthetic_dict[sta_code]['S'].time

        found = False
        for sta in stations:
            if sta.code == sta_code:
                d['loc'] = sta.loc
                found = True
                break
        if not found:
            print("Oops: sta:%s not found in inventory!" % sta_code)
            raise

    # TODO May want to revise to use distance along raypath here
        d['R'] = np.linalg.norm(d['loc'] - origin.loc) # Dist in meters
        sta_dict[sta_code] = d


# 2. Calc/save signal/noise fft spectra at all channels that have P arrivals:
    for sta_code,sta in sta_dict.items():

        if P_or_S == 'P':
            signal_start = sta['ptime'] - .01
            signal_end   = sta['stime'] - .01
        else:
            signal_start = sta['stime'] - .01
            signal_end   = sta['stime'] + S_win_len

        signal_len   = signal_end - signal_start

        noise_end   = sta['ptime'] - .01
        noise_start = noise_end - signal_len

        trs = st.select(station=sta_code)
        if trs:
            chans = {}

            for tr in trs:
                ch = {}

                tt_s = sta['stime'] - tr.stats.starttime
                tr.detrend('demean').detrend('linear').taper(type='cosine', max_percentage=0.05, side='both')

                signal = tr.copy()
                signal.trim(starttime=signal_start, endtime=signal_end)
                signal.detrend('demean').detrend('linear')
                signal.taper(type='cosine', max_percentage=0.05, side='both')

                # If this trace sensor_type='ACCELEROMETER' --> integrate to velocity
                if sta_meta_dict[tr.stats.station][tr.stats.channel].sensor_type == 'ACCELEROMETER':
                    signal.integrate().taper(type='cosine', max_percentage=0.05, side='both')

                if calc_displacement:
                    signal.integrate().taper(type='cosine', max_percentage=0.05, side='both')

                noise  = tr.copy()
            # if noise_start < trace.stats.starttime - then what ?
                noise.trim(starttime=noise_start, endtime=noise_end)
                noise.detrend('demean').detrend('linear')
                noise.taper(type='cosine', max_percentage=0.05, side='both')

                # If this trace sensor_type='ACCELEROMETER' --> integrate to velocity
                if sta_meta_dict[tr.stats.station][tr.stats.channel].sensor_type == 'ACCELEROMETER':
                    signal.integrate().taper(type='cosine', max_percentage=0.05, side='both')

                if calc_displacement:
                    noise.integrate().taper(type='cosine', max_percentage=0.05, side='both')

                #check(signal)
                parsevals(signal.data, dt, nfft)

                (signal_fft, freqs) = unpack_rfft( rfft(signal.data, n=nfft), df)
                (noise_fft, freqs)  = unpack_rfft( rfft(noise.data, n=nfft), df)

                #plot_signal(signal, noise)
                #plot_spec(freqs, signal_fft, noise_fft, title=None)
                #exit()

            # np/scipy ffts are not scaled by dt
            # Do it here so we don't forget 
                signal_fft *= dt
                noise_fft  *= dt

                #plot_signal(signal, noise)

                ch['nfft'] = nfft
                ch['dt']   = dt
                ch['df']   = df
                ch['freqs'] = freqs
                ch['signal_fft'] = signal_fft
                ch['noise_fft']  = noise_fft
                if calc_displacement:
                    ch['disp_or_vel'] = 'DISPLACEMENT'
                else:
                    ch['disp_or_vel'] = 'VELOCITY'
                chans[tr.stats.channel] = ch
            sta['chan_spec'] = chans


    return sta_dict


def brune_dis_spec(fc: float, mom: float, ts: float, f: float) -> float :
    return mom * (fc * fc) / (fc * fc + f * f) * np.exp(-np.pi * ts * f)

def brune_vel_spec(fc: float, mom: float, ts: float, f: float) -> float :
    return 2. * np.pi * f * brune_dis_spec(fc, mom, ts, f)

def brune_acc_spec(fc: float, mom: float, ts: float, f: float) -> float :
    return 2. * np.pi * f * brune_vel_spec(fc, mom, ts, f)


def getresidfit(data_spec, model_func, fc: float, freqs: list, Lnorm='L1', weight_fr=False, fmin=1., fmax=3000.) -> float :

    def inner(p):
        smom=p[0]
        ts  =p[1]
        fit = 0.
        # Neither Powell nor Nelder-Meade seem to have a way to restrict search params to be > 0,
        # and L-BFGS-B does not appear to work, so here's a hack:
        if smom < 0. or ts < 0.:
            return 1e12

        for i,f in enumerate(freqs):
            if f < fmin or f > fmax:
                continue
            if model_func(fc, smom, ts, f) < 0.:
                print("** OOPS: f=%f vel_spec=%g smom=%g ts=%g model_func=%g" % (f, vel_spec[i], smom, ts, model_func(fc,smom,ts,f)))
                pass
            diff = np.log10(data_spec[i]) - np.log10(model_func(fc, smom, ts, f))
            if Lnorm=='L2':
                diff *= diff
            if weight_fr:
                diff /= f
            fit += diff
        return fit
    return inner

def getresidfit2(data_spec, model_func, freqs: list, Lnorm='L1', weight_fr=False, fmin=1., fmax=3000.) -> float :

    def inner(p):
        smom=p[0]
        fc  =p[1]
        fit = 0.
        # Neither Powell nor Nelder-Meade seem to have a way to restrict search params to be > 0,
        # and L-BFGS-B does not appear to work, so here's a hack:
        if smom < 0. :
            return 1e12

        for i,f in enumerate(freqs):
            if f < fmin or f > fmax:
                continue
            diff = np.log10(data_spec[i]) - np.log10(model_func(fc, smom, f))
            if Lnorm=='L2':
                diff *= diff
            if weight_fr:
                diff /= f
            fit += diff
        return fit
    return inner

def getresidfit3(data_spec, model_func, freqs: list, Lnorm='L1', weight_fr=False, fmin=1., fmax=3000.) -> float :

    def inner(p):
        smom=p[0]
        ts  =p[1]
        fc  =p[2]
        fit = 0.
        # Neither Powell nor Nelder-Meade seem to have a way to restrict search params to be > 0,
        # and L-BFGS-B does not appear to work, so here's a hack:
        if smom < 0. or ts < 0.:
            return 1e12

        for i,f in enumerate(freqs):
            if f < fmin or f > fmax:
                continue
            if model_func(fc, smom, ts, f) < 0.:
                print("** OOPS: f=%f vel_spec=%g smom=%g ts=%g model_func=%g" % (f, vel_spec[i], smom, ts, model_func(fc,smom,ts,f)))
                pass
            diff = np.log10(data_spec[i]) - np.log10(model_func(fc, smom, ts, f))
            if Lnorm=='L2':
                diff *= diff
            if weight_fr:
                diff /= f
            fit += diff
        return fit
    return inner


def calc_fit3(spec, freqs, fmin=1., fmax=1000., Lnorm='L2', weight_fr=False, fit_displacement=False):

    model_func = brune_vel_spec
    if fit_displacement:
        model_func = brune_dis_spec

    residfit = getresidfit3(spec, model_func, freqs, Lnorm=Lnorm, \
                            weight_fr=weight_fr, fmin=fmin, fmax=fmax)

    # Give it a starting vector:
    #    smom  t* fc
    pp = [1., .001, 100]
    (sol,fit,*rest)= optimize.fmin(residfit, np.array(pp),xtol=10**-12,ftol=10**-6,disp=False, full_output=1)
    mom, ts, fc = sol[0], sol[1], sol[2]
    if sol[0] < 0. :
        print("Ummm smom < 0 !!!")

    print("solution=",sol)
    plot_fit = 1
    if plot_fit:
        model_spec = np.array( np.zeros(freqs.size), dtype=np.float_)
        for i,f in enumerate(freqs):
            model_spec[i] = model_func(fc, mom, ts, f)
        plot_spec2(freqs, spec, model_spec, title="Fit to spec stack fc=%f" % fc)
    return fit

def calc_fit2(spec, freqs, fmin=10., fmax=1000., Lnorm='L2', weight_fr=False):

    model_func = brune_vel_spec2

    residfit = getresidfit2(spec, model_func, freqs, Lnorm=Lnorm, \
                            weight_fr=weight_fr, fmin=fmin, fmax=fmax)

    # Give it a starting vector:
    pp = [1., 100]
    (sol,fit,*rest)= optimize.fmin(residfit, np.array(pp),xtol=10**-12,ftol=10**-6,disp=False, full_output=1)
    if sol[0] < 0. :
        print("Ummm smom < 0 !!!")

    print("solution=",sol)
    plot_fit = 1
    if plot_fit:
        model_spec = np.array( np.zeros(freqs.size), dtype=np.float_)
        for i,f in enumerate(freqs):
            model_spec[i] = model_func(100., 1., f)
            #model_spec[i] = model_func(80., sol[0], f)
            #model_spec[i] = model_func(sol[1], sol[0], f)
        plot_spec2(freqs, spec, model_spec, title="Fit to spec stack fc=%f" % sol[1])
    return fit

def calc_fit(sta_dict, fc, fmin=20., fmax=1000., Lnorm='L2', weight_fr=False):

    fit = 0.
    smom_dict = {}
    for sta_code, sta in sta_dict.items():
        #print("  calc_fit sta:%3s" % sta_code)
        if 'chan_spec' not in sta:
            print("  sta:%3s: Has no chan_spec --> Skip" % sta_code)
            continue
        dd  = {}
        for cha_code, cha in sta['chan_spec'].items():
            #continue
            #print("    calc_fit Process: cha:%s" % cha_code)
            signal_fft, freqs, noise_fft = (cha['signal_fft'], cha['freqs'], cha['noise_fft'])

            model_func = brune_vel_spec
            if cha['disp_or_vel'] == 'DISPLACEMENT':
                model_func = brune_dis_spec

            ch_dict = {}
            # Optimize fit of model_func to signal_fft to solve for smom:

            residfit = getresidfit(np.abs(signal_fft), model_func, fc, freqs, Lnorm=Lnorm, \
                                           weight_fr=weight_fr, fmin=fmin, fmax=fmax)

            # Give it a starting vector:
            #     smom    tstar
            # MTH: this should be wert origin time anyway ...
            #tt_s = sta['stime'] - tr.stats.starttime
            tt_s = .02
            #pp = [1e-5, tt_s/500.]
            #pp = [1e-5, tt_s/200.]
            pp = [1e-10, tt_s/200.]
            #(sol,fopt,*rest)= optimize.fmin(residfit, np.array(pp),xtol=10**-12,ftol=10**-12,disp=False, full_output=1)
            #(sol,fopt,*rest)= optimize.fmin(residfit, np.array(pp),xtol=10**-3,ftol=10**-3,disp=False, full_output=1)
            (sol,fopt,*rest)= optimize.fmin(residfit, np.array(pp),ftol=.01, disp=False, full_output=1)
            #(sol,fopt,*rest)= optimize.fmin(residfit, np.array(pp),xtol=10**-3,ftol=10**-3,disp=False, full_output=1)
            #(sol,fopt,*rest)= optimize.fmin(residfit, np.array(pp),xtol=10**-6,ftol=10**-6,disp=False, full_output=1)
            if sol[0] < 0. :
                print("Ummm smom < 0 !!!")

            #gives same result as fopt
            #ch_fit = residfit(sol)
            fit += fopt

            ch_dict['smom'] = sol[0]
            ch_dict['fit'] = fopt
            #print("    cha:%s smom:%12.10g" % (cha_code, sol[0]))

            #print("calc_fit: sta:%s cha:%s smom:%12.10g fit:%.2f" % (sta_code, cha_code, sol[0], fopt))
            plot_fit = 1
            plot_fit = 0
            if plot_fit:
                model_spec = np.array( np.zeros(freqs.size), dtype=np.float_)
                for i,f in enumerate(freqs):
                    model_spec[i] = model_func(fc, sol[0], sol[1], f)
                #plot_spec(freqs, np.abs(signal_fft),  model_spec, title="sta:%s ch:%s spec fit" % \
                plot_spec3(freqs, np.abs(signal_fft),  np.abs(noise_fft), model_spec, title="sta:%s ch:%s spec fit" % \
                          (sta_code, cha_code))

            dd[cha_code] = ch_dict
        smom_dict[sta_code] = dd

    return fit, smom_dict

def plot_spec3(freqs, signal_fft, noise_fft, model_spec, title=None):

    plt.loglog(freqs, np.abs(signal_fft), color='blue')
    plt.loglog(freqs, np.abs(noise_fft),  color='red')
    plt.loglog(freqs, model_spec,  color='green')
    plt.legend(['signal', 'noise', 'model'])
    plt.xlim(1e0, 3e3)
    plt.ylim(1e-12, 1e-6)
    if title:
        plt.title(title)
    plt.grid()
    plt.show()

#def plot_spec(freqs, signal_fft, noise_fft, model_spec, title=None):
def plot_spec(freqs, signal_fft, noise_fft=None, title=None):

    plt.loglog(freqs, np.abs(signal_fft), color='blue')
    if noise_fft:
        plt.loglog(freqs, np.abs(noise_fft),  color='red')
    #plt.loglog(freqs, model_spec,  color='green')
    #plt.legend(['signal', 'noise', 'model'])
    if noise_fft:
        plt.legend(['signal', 'noise'])
    plt.xlim(1e0, 3e3)
    plt.ylim(1e-8, 1.2e-4)
    #plt.ylim(1e-12, 1e-4)
    if title:
        plt.title(title)
    plt.grid()
    plt.show()

def plot_spec2(freqs, spec, model_spec, title=None):

    plt.loglog(freqs, spec, color='blue')
    plt.loglog(freqs, model_spec,  color='green')
    plt.legend(['signal', 'model'])
    plt.xlim(1e0, 3e3)
    #plt.ylim(1e-12, 1e-4)
    if title:
        plt.title(title)
    plt.grid()
    plt.show()


def plot_signal(signal, noise=None):
#def plot_signal(signal, noise):
    '''
        signal = obspy Trace windowed around signal
        noise  = obspy Trace windowed around pre-signal noise
    '''
    t = []
    dt = signal.stats.sampling_rate
    for i in range(signal.stats.npts ):
        t.append(float(i)*dt)
    plt.plot(t, signal, color='blue')
    if noise:
        plt.plot(t, noise, color='red')
        plt.legend(['signal', 'noise'])
    plt.title("%s: P window" % signal.get_id())
    plt.grid()
    plt.show()


def unpack_rfft(rfft, df):
    n = rfft.size
    if n % 2 == 0:
        n2 = int(n/2)
    else:
        print("n is odd!!")
        exit()
    #print("n2=%d" % n2)

    c_arr = np.array( np.zeros(n2+1,), dtype=np.complex_)
    freqs = np.array( np.zeros(n2+1,), dtype=np.float_)

    c_arr[0]  = rfft[0]
    c_arr[n2] = rfft[n-1]
    freqs[0]  = 0.
    freqs[n2] = float(n2)*df

    for i in range(1, n2):
        freqs[i] = float(i)*df
        c_arr[i] = np.complex(rfft[2*i - 1], rfft[2*i])

    return c_arr, freqs


import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
        #raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
        #raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        #raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


def npow2(n: int) -> int:
    """ return power of 2 >= n
    """
    if n <= 0:
        return 0

    nfft = 2
    while (nfft < n):
        nfft = (nfft << 1)
    return nfft

if __name__ == '__main__':

    main()
