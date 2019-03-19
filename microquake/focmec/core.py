
import numpy as np

#from hashwrap.hashwrapper import calc_focal_mechanisms as calc_HASH_focal_mechanisms

from hashwrap import hashwrapper

from obspy.core.event.source import FocalMechanism, NodalPlane, NodalPlanes
from obspy.imaging.beachball import aux_plane
from obspy.core.event.base import Comment

import logging
logger = logging.getLogger(__name__)

def calc_focal_mechanisms(cat, settings, logger_in=None):
    """
    Prepare input arrays needed to calculate focal mechanisms
    and pass these into hashwrap.hashwrapper

    Return list of obspy focalmechanisms & list of matplotlib figs 

    :param cat: obspy.core.event.Catalog
    :type list: list of obspy.core.event.Events or microquake.core.event.Events
    :param settings:hash settings
    :type settings dictionary
    :param logger_in: Optional logger to use instead of default module logger
    :type logger_in: Handle to logger object

    :returns: obsy_focal_mechanisms, matplotlib_figures
    :rtype: list, list
    """

    fname = 'calc_focal_mechanism'

    global logger
    if logger_in is not None:
        logger = logger_in

    sname  = []
    p_pol  = []
    p_qual = []
    qdist = []
    qazi = []
    qthe = []
    sazi = []
    sthe = []


    events = []

    for event in cat:

        event_dict = {}

        origin = event.preferred_origin()

        event_dict['event_info'] = origin.time.datetime.strftime('%Y-%m-%d %H:%M:%S')
        event_dict['event'] = {}
        event_dict['event']['qdep']= origin.loc[2]
        event_dict['event']['sez']= 10.
        event_dict['event']['icusp']= 1234567

        arrivals = [arr for arr in event.preferred_origin().arrivals if arr.phase == 'P']

        for arr in arrivals:

            if arr.pulse_snr is None:
                logger.warn("%s P arr pulse_snr == NONE !!!" % \
                      arr.pick_id.get_referred_object().waveform_id.station_code)
                continue

            sname.append(arr.pick_id.get_referred_object().waveform_id.station_code)
            p_pol.append(arr.polarity)
            qdist.append(arr.distance)
            qazi.append(arr.azimuth)
    # MTH: both HASH and test_stereo expect takeoff theta measured wrt vertical Up!
            qthe.append(180. - arr.takeoff_angle)
            sazi.append(2.)
            sthe.append(10.)

            if arr.pulse_snr >= 100.:
                qual = 0
            else:
                qual = 1
            p_qual.append(qual)

        event_dict['sname'] = sname
        event_dict['p_pol'] = p_pol
        event_dict['p_qual'] = p_qual
        event_dict['qdist'] = qdist
        event_dict['qazi'] = qazi
        event_dict['qthe'] = qthe
        event_dict['sazi'] = sazi
        event_dict['sthe'] = sthe

    events.append(event_dict)

    outputs = hashwrapper.calc_focal_mechanisms(events, settings,
                                                phase_format='FPFIT',
                                                logger_in=logger)

    focal_mechanisms = []

    plot_figures = []
    for i,out in enumerate(outputs):
        logger.info("%s.%s: Process Focal Mech i=%d" % (__name__,fname, i))
        p1 = NodalPlane(strike=out['strike'], dip=out['dip'], rake=out['rake'])
        s,d,r = aux_plane(out['strike'], out['dip'], out['rake'])
        p2 = NodalPlane(strike=s, dip=d, rake=r)

        fc = FocalMechanism(nodal_planes = NodalPlanes(nodal_plane_1=p1, nodal_plane_2=p2),
                            azimuthal_gap = out['azim_gap'],
                            station_polarity_count = out['station_polarity_count'],
                            station_distribution_ratio = out['stdr'],
                            misfit = out['misfit'],
                            evaluation_mode = 'automatic',
                            evaluation_status = 'preliminary',
                            comments = [Comment(text="HASH v1.2 Quality=[%s]" % out['quality'])]
                           )

        focal_mechanisms.append(fc)

        event = events[i]

        title = "%s (s,d,r)_1=(%.1f,%.1f,%.1f) _2=(%.1f,%.1f,%.1f)" % \
                (event['event_info'], p1.strike, p1.dip, p1.rake, p2.strike,p2.dip,p2.rake)

        gcf = test_stereo(np.array(event['qazi']),np.array(event['qthe']),np.array(event['p_pol']),\
                    sdr=[p1.strike,p1.dip,p1.rake], title=title)

        plot_figures.append(gcf)

    return focal_mechanisms, plot_figures


import matplotlib.pyplot as plt
import mplstereonet
from obspy.imaging.beachball import aux_plane

def test_stereo(azimuths,takeoffs,polarities,sdr=[], title=None):
    '''
        Plots points with given azimuths, takeoff angles, and
        polarities on a stereonet. Will also plot both planes
        of a double-couple given a strike/dip/rake
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')
    up = polarities > 0
    dn = polarities < 0
    #h_rk = ax.rake(azimuths[up]-90.,takeoffs[up],90, 'ro')
    # MTH: this seems to put the observations in the right location
    #  We're plotting a lower-hemisphere focal mech, and we have to convert
    #  the up-going rays to the right az/dip quadrants:
    h_rk = ax.rake(azimuths[up]-90.+180.,90.-takeoffs[up],90, 'ro')
    h_rk = ax.rake(azimuths[dn]-90.+180.,90.-takeoffs[dn],90, 'b+')
    #ax.rake(strike-90., 90.-dip, rake, 'ro', markersize=14)

    #h_rk = ax.rake(azimuths[dn]-90.,takeoffs[dn],90, 'b+')
    if sdr:
        s2,d2,r2 = aux_plane(*sdr)
        h_rk = ax.plane(sdr[0],sdr[1],'g')
        h_rk = ax.rake(sdr[0],sdr[1],-sdr[2], 'go')
        h_rk = ax.plane(s2,d2, 'g')

    if title:
        plt.title(title)

    #plt.show()

    return plt.gcf()


if __name__ == '__main__':
    main()
