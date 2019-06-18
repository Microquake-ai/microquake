# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
#  Purpose: module to interact with the NLLoc
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
module to interact with the NLLoc

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


# from microquake.nlloc.core import *

from datetime import datetime
import shutil
import os
import tempfile
import numpy as np
from microquake.core.event import Catalog, Arrival, Origin
from microquake.core.util.attribdict import AttribDict
from microquake.core import UTCDateTime
import logging
from glob import glob

logger = logging.getLogger()


def read_nlloc_hypocenter_file(filename, picks=None,
                               evaluation_mode='automatic',
                               evaluation_status='preliminary'):
    """
    read NLLoc hypocenter file into an events catalog
    :param filename: path to NLLoc hypocenter filename
    :type filename: str
    :return: seismic catalogue
    :rtype: ~microquake.core.event.Catalog
    """
    from microquake.core import event
    cat = event.Catalog()

    with open(filename) as hyp_file:

        #print("read_nlloc_file:%s" % filename)
        #exit()
        all_lines = hyp_file.readlines()
        hyp = [line.split() for line in all_lines if 'HYPOCENTER' in line][0]
        stat = [line.split() for line in all_lines if 'STATISTICS' in line][0]
        geo = [line.split() for line in all_lines if 'GEOGRAPHIC' in line][0]
        qual = [line.split() for line in all_lines if 'QUALITY' in line][0]
        search = [line.split() for line in all_lines if 'SEARCH' in line][0]
        sign = [line.split() for line in all_lines if 'SIGNATURE' in line][0]

        s = int(np.floor(float(geo[7])))
        us = int((float(geo[7]) - s) * 1e6)
        if s < 0:
            s = 0
        if us < 0:
            us = 0

        tme = datetime(int(geo[2]), int(geo[3]), int(geo[4]),
                       int(geo[5]), int(geo[6]), s, us)
        tme = UTCDateTime(tme)

        if 'REJECTED' in all_lines[0]:
            evaluation_status = 'rejected'
            logger.warning('Event located on grid boundary')
        else:
            evaluation_status = evaluation_status

        hyp_x = float(hyp[2]) * 1000
        hyp_y = float(hyp[4]) * 1000
        hyp_z = float(hyp[6]) * 1000

        method = '%s' % ("NLLOC")

        creation_info = event.CreationInfo(author='microquake',
                                           creation_time=UTCDateTime.now())

        origin = event.Origin(x=hyp_x, y=hyp_y, z=hyp_z, time=tme,
                              evaluation_mode=evaluation_mode,
                              evaluation_status=evaluation_status,
                              epicenter_fixed=0, method_id=method,
                              creation_info=creation_info)

        xminor = np.cos(float(stat[22]) * np.pi / 180) * np.sin(float(stat[20])
                                                                * np.pi / 180)
        yminor = np.cos(float(stat[22]) * np.pi / 180) * np.cos(float(stat[20])
                                                                * np.pi / 180)
        zminor = np.sin(float(stat[22]) * np.pi / 180)
        xinter = np.cos(float(stat[28]) * np.pi / 180) * np.sin(float(stat[26])
                                                                * np.pi / 180)
        yinter = np.cos(float(stat[28]) * np.pi / 180) * np.cos(float(stat[26])
                                                                * np.pi / 180)
        zinter = np.sin(float(stat[28]) * np.pi / 180)

        minor = np.array([xminor, yminor, zminor])
        inter = np.array([xinter, yinter, zinter])

        major = np.cross(minor, inter)

        major_az = np.arctan2(major[0], major[1])
        major_dip = np.arctan(major[2] / np.linalg.norm(major[0:2]))
        # MTH: obspy will raise error if you try to set float attr to nan below
        if np.isnan(major_az):
            major_az = None
        if np.isnan(major_dip):
            major_dip = None

        ou = event.OriginUncertainty()
        el = event.ConfidenceEllipsoid()
        el.semi_minor_axis_length = float(stat[24]) * 1000
        el.semi_intermediate_axis_length = float(stat[30]) * 1000
        el.semi_major_axis_length = float(stat[32]) * 1000
        el.major_axis_azimuth = major_az
        el.major_axis_plunge = major_dip

        ou.confidence_ellipsoid = el

        origin.origin_uncertainty = ou

        TravelTime = False
        oq = event.OriginQuality()
        arrivals = []
        stations = []
        phases = []
        oq.associated_phase_count = 0
        for line in all_lines:
            if 'PHASE ' in line:
                TravelTime = True
                continue
            elif 'END_PHASE' in line:
                TravelTime = False
                continue

            if TravelTime:
                tmp = line.split()
                stname = tmp[0]

                phase = tmp[4]
                res = float(tmp[16])
                weight = float(tmp[17])
                sx = float(tmp[18])
                sy = float(tmp[19])
                sz = float(tmp[20])
    # MTH: In order to not get default = -1.0 for ray azimuth + takeoff here, you
    #      need to set ANGLES_YES in the NLLOC Grid2Time control file. Then, when Grid2Time runs, it
    #      will also create the angle.buf files in NLLOC/run/time and when NLLoc runs, it will interpolate
    #      these to get the ray azi + takeoff and put them on the phase line of last.hyp
    # However, the NLLoc generated takeoff angles look to be incorrect (< 90 deg),
    #  likely due to how OT vertical up convention wrt NLLoc.
    # So instead, use the spp generated files *.azimuth.buf and *.takeoff.buf to overwrite these later
    #      15       16       17              18  19       20          21       22     23 24
    #  >   TTpred    Res       Weight    StaLoc(X  Y         Z)        SDist    SAzim  RAz  RDip RQual    Tcorr
    #  >  0.209032  0.002185    1.2627  651.3046 4767.1881    0.9230    0.2578 150.58  -1.0  -1.0  0     0.0000


                azi = float(tmp[22]) # Set to SAzim since that is guaranteed to be set
                #azi = float(tmp[23])
                toa = float(tmp[24])

                dist = np.linalg.norm([sx * 1000 - origin.x,
                                       sy * 1000 - origin.y,
                                       sz * 1000 - origin.z])

                '''
                MTH: Some notes about the NLLOC output last.hyp phase lines:
                    1. SDist - Is just epicentral distance so does not take into account dz (depth)
                               So 3D Euclidean dist as calculated above will be (much) larger
                    2. SAzim - NLLOC manual says this is measured from hypocenter CCW to station
                               But it looks to me like it's actually clockwise!
                    3. RAz - "Ray take−off azimuth at maximum likelihood hypocenter in degrees CCW from North"
                              In a true 3D model (e.g., lateral heterogeneity) this could be different
                              than SAzim.
                              Have to set: LOCANGLES ANGLES_YES 5 to get the angles, otherwise defaults to -1
                              Probably these are also actually measured clockwise from North

                distxy = np.linalg.norm([sx * 1000 - origin.x,
                                         sy * 1000 - origin.y])

                sdist = float(tmp[21])
                sazim = float(tmp[22])
                raz = float(tmp[23])
                rdip = float(tmp[24])

                print("Scan last.hyp: sta:%3s pha:%s dist_calc:%.1f sdist:%.1f sazim:%.1f raz:%.1f rdip:%.1f" % \
                      (stname, phase, distxy, sdist*1e3, sazim, raz, rdip))

                '''

                arrival = Arrival()
                arrival.phase = phase
                arrival.distance = dist
                arrival.time_residual = res
                arrival.time_weight = weight
                arrival.azimuth = azi
                arrival.takeoff_angle = toa
                arrivals.append(arrival)

                for pick in picks:
                    if ((pick.phase_hint == phase) and (
                            pick.waveform_id.station_code == stname)):

                        arrival.pick_id = pick.resource_id.id

                stations.append(stname)
                phases.append(phase)

                oq.associated_phase_count += 1

        stations = np.array(stations)

        points = read_scatter_file(filename.replace('.hyp','.scat'))

        origin.arrivals = [arr for arr in arrivals]
        origin.scatter = points

        oq.associated_station_count = len(np.unique(stations))

        oq.used_phase_count = oq.associated_phase_count
        oq.used_station_count = oq.associated_station_count
        oq.standard_error = float(qual[8])
        oq.azimuthal_gap = float(qual[12])
        origin.quality = oq

    return origin


def calculate_uncertainty(event, base_directory, base_name, perturbation=5,
                          pick_uncertainty=1e-3):
        """
        :param cat: event
        :type cat: microquake.core.event.Event
        :param base_directory: base directory
        :param base_name: base name for grids
        :param perturbation:
        :return: microquake.core.event.Event
        """

        from microquake.core.data.grid import read_grid
        from numpy import linalg, argsort, arcsin, sqrt
        from microquake.core.event import ConfidenceEllipsoid, OriginUncertainty

        narr = len(event.preferred_origin().arrivals)

        # initializing the frechet derivative
        Frechet = np.zeros([narr, 3])

        event_loc = np.array(event.preferred_origin().loc)

        for i, arrival in enumerate(event.preferred_origin().arrivals):
            pick = arrival.pick_id.get_referred_object()
            station_code = pick.waveform_id.station_code
            phase = arrival.phase

            # loading the travel time grid
            filename = '%s/time/%s.%s.%s.time' % (base_directory,
            base_name, phase, station_code)

            tt = read_grid(filename, format='NLLOC')
            spc = tt.spacing

            # build the Frechet derivative
            for dim in range(0,3):
                loc_p1 = event_loc.copy()
                loc_p2 = event_loc.copy()
                loc_p1[dim] += perturbation
                loc_p2[dim] -= perturbation
                tt_p1 = tt.interpolate(loc_p1, grid_coordinate=False)
                tt_p2 = tt.interpolate(loc_p2, grid_coordinate=False)
                Frechet[i, dim] = (tt_p1 - tt_p2) / (2 * perturbation)

        hessian = linalg.inv(np.dot(Frechet.T, Frechet))
        tmp = hessian * pick_uncertainty ** 2
        w, v = linalg.eig(tmp)
        i = argsort(w)[-1::-1]
        # for the angle calculation see
        # https://en.wikipedia.org/wiki/Euler_angles#Tait-Bryan_angles
        X = v[:, i[0]]
        Y = v[:, i[1]]
        Z = v[:, i[2]]
        # Tracer()()
        major_axis_plunge = arcsin(Y[2] / sqrt(1 - X[2] ** 2))
        major_axis_azimuth = arcsin(X[1] / sqrt(1 - X[2] ** 2))
        major_axis_rotation = arcsin(-X[2])
        ce = ConfidenceEllipsoid(semi_major_axis_length=w[i[0]],
                                 semi_intermediate_axis_length=w[i[1]],
                                 semi_minor_axis_length=w[i[2]],
                                 major_axis_plunge=major_axis_plunge,
                                 major_axis_azimuth=major_axis_azimuth,
                                 major_axis_rotation=major_axis_rotation)
        ou = OriginUncertainty(confidence_ellipsoid=ce)

        return ou


def read_scatter_file(filename):
    """
    :param filename: name of the scatter file to read
    :return: a numpy array of the points in the scatter file
    """

    import struct
    from numpy import array

    f = open(filename, 'rb')

    nsamples = struct.unpack('i', f.read(4))[0]
    struct.unpack('f', f.read(4))
    struct.unpack('f', f.read(4))
    struct.unpack('f', f.read(4))

    points = []
    for k in range(0, nsamples):
        x = struct.unpack('f', f.read(4))[0]
        y = struct.unpack('f', f.read(4))[0]
        z = struct.unpack('f', f.read(4))[0]
        pdf = struct.unpack('f', f.read(4))[0]

        points.append([x, y, z, pdf])

    return array(points)


def is_supported_nlloc_grid_type(grid_type):
    """
    verify that the grid_type is a valid NLLoc grid type
    :param grid_type: grid_type
    :type grid_type: str
    :rtype: bool
    """
    grid_type = grid_type.upper()

    if grid_type in supported_nlloc_grid_type:
        return True

    return False


def _read_nll_header_file(file_name):
    """
    read NLLoc header file
    :param file_name: path to the header file
    :type file_name: str
    :rtype: ~microquake.core.util.attribdict.AttribDict
    """
    dict_out = AttribDict()
    with open(file_name, 'r') as fin:
        line = fin.readline()
        line = line.split()
        dict_out.shape = tuple([int(line[0]), int(line[1]), int(line[2])])
        dict_out.origin = np.array([float(line[3]), float(line[4]),
                                    float(line[5])])
        dict_out.origin *= 1000
        dict_out.spacing = float(line[6]) * 1000
        dict_out.grid_type = line[9]

        line = fin.readline()
        if dict_out.grid_type in ['ANGLE', 'TIME']:
            line = line.split()
            dict_out.label = line[0]
            dict_out.seed = (float(line[1]) * 1000,
                             float(line[2]) * 1000,
                             float(line[3]) * 1000)

        else:
            dict_out.label = None
            dict_out.seed = None

    return dict_out


def read_NLL_grid(base_name):
    """
    read NLL grids into a GridData object
    :param base_name: path to the file excluding the extension. The .hdr and
    .buf extensions are added automatically
    :type base_name: str
    :rtype: ~microquake.core.data.grid.GridDataa

    .. NOTE:
        The function detects the presence of either the .buf or .hdr extensions
    """

    from microquake.core import GridData
    from glob import glob
    # Testing the presence of the .buf or .hdr extension at the end of base_name
    if ('.buf' == base_name[-4:]) or ('.hdr' == base_name[-4:]):
        # removing the extension
        base_name = base_name[:-4]

    # Reading header file
    try:
        head = _read_nll_header_file(base_name + '.hdr')
    except ValueError:
        logger.error('error reading %s' % base_name + '.hdr')

    # Read binary buffer
    gdata = np.fromfile(base_name + '.buf', dtype=np.float32)
    gdata = gdata.reshape(head.shape)
    if head.grid_type == 'SLOW_LEN':
        gdata = head.spacing / gdata
        head.grid_type = 'VELOCITY'


    return GridData(gdata, spacing=head.spacing, origin=head.origin,
                    seed=head.seed, seed_label=head.label,
                    grid_type=head.grid_type)


def _write_grid_data(base_name, data):
    """
    write 3D grid data to a NLLoc grid
    :param base_name: file name without the extension (.buf extension will be
    added automatically)
    :type base_name: str
    :param data: 3D grid data to be written
    :type data: 3D numpy.array
    :rtype: None
    """
    with open(base_name + '.buf', 'wb') as ofile:
        ofile.write(data.astype(np.float32).tobytes())


def _write_grid_header(base_name, shape, origin, spacing, grid_type,
                       station=None, seed=None):
    """
    write NLLoc grid header file
    :param base_name: file name without the extension (.buf extension will be
    added automatically)
    :type base_name: str
    :param shape: grid shape
    :type shape: tuple, list or numpy.array
    :param origin: grid origin
    :type origin: tuple, list or numpy.array
    :param spacing: grid spacing
    :type spacing: float
    :param grid_type: type of NLLoc grid. For valid choice see below. Note that
    the grid_type is not case sensitive (e.g., 'velocity' == 'VELOCITY')
    :type grid_type: str
    :param station: station code or name (required only for certain grid type)
    :type station: str
    :param seed: the station location (required only for certain grid type)
    :type seed: tuple, list or numpy.array

    """

    line1 = u"%d %d %d  %f %f %f  %f %f %f  %s\n" % (
            shape[0], shape[1], shape[2],
            origin[0] / 1000., origin[1] / 1000., origin[2] / 1000.,
            spacing / 1000., spacing / 1000., spacing / 1000.,
            grid_type)

    with open(base_name + '.hdr', 'w') as ofile:
        ofile.write(line1)

        if grid_type in ['TIME', 'ANGLE']:
            line2 = u"%s %f %f %f\n" % (station, seed[0], seed[1], seed[2])
            ofile.write(line2)

        ofile.write(u'TRANSFORM  NONE\n')

    return


def write_nll_grid(base_name, data, origin, spacing, grid_type, seed=None,
                   label=None, velocity_to_slow_len=True):
    """
    Write write structure data grid to NLLoc grid format
    :param base_name: output file name and path without extension
    :type base_name: str
    :param data: structured data
    :type data: numpy.ndarray
    :param origin: grid origin
    :type origin: tuple
    :param spacing: spacing between grid nodes (same in all dimensions)
    :type spacing: float
    :param grid_type: type of grid (must be a valid NLL grid type)
    :type grid_type: str
    :param seed: seed of the grid value. Only required / used for "TIME" or
    "ANGLE" grids
    :type seed: tuple
    :param label: seed label (usually station code). Only required / used for
    "TIME" and "ANGLE" grids
    :type label: str
    :param velocity_to_slow_len: convert "VELOCITY" to "SLOW_LEN". NLLoc
    Grid2Time program requires that "VELOCITY" be expressed in "SLOW_LEN" units.
    Has influence only if the grid_type is "VELOCITY"
    :type velocity_to_slow_len: bool
    :rtype: None

    supported NLLoc grid types are

    "VELOCITY": velocity (km/sec);
    "VELOCITY_METERS": velocity (m/sec);
    "SLOWNESS = slowness (sec/km);
    "SLOW_LEN" = slowness*length (sec);
    "TIME" = time (sec) 3D grid;
    "PROB_DENSITY" = probability density;
    "MISFIT" = misfit (sec);
    "ANGLE" = take-off angles 3D grid;
    """

    if not is_supported_nlloc_grid_type(grid_type):
        logger.warning('Grid type is not a valid NLLoc type')

    # removing the extension if extension is part of the base name
    if ('.buf' == base_name[-4:]) or ('.hdr' == base_name[-4:]):
        # removing the extension
        base_name = base_name[:-4]

    if (grid_type == 'VELOCITY') and (velocity_to_slow_len):
        tmp_data = spacing / data  # need this to be in SLOW_LEN format (s/km2)
        grid_type = 'SLOW_LEN'
    else:
        tmp_data = data

    _write_grid_data(base_name, tmp_data)

    shape = data.shape

    _write_grid_header(base_name, shape, origin, spacing,
                       grid_type, label, seed)




# def prepare_nll(ctl_filename='input.xml', nll_base='NLL'):
#     """
#     :param ctl_filename: path to the XML file containing control parameters
#     :param nll_base: directory in which NLL project will be built
#     """
#     params = ctl.parseControlFile(ctl_filename)
#     keys = ['velgrids', 'sensors']
#     for job_index, job in enumerate(ctl.buildJob(keys, params)):
#
#         params = ctl.getCurrentJobParams(params, keys, job)
#         nll_opts = init_from_xml_params(params, base_folder=nll_base)
#         nll_opts.prepare(create_time_grids=True, tar_files=False)


def init_nlloc_from_params(params):
    """

    """
    project_code = params.project_code

    nll = NLL(project_code, base_folder=params.nll.NLL_BASE)
    nll.gridpar = params.velgrids
    nll.sensors = params.sensors
    nll.params = params.nll

    nll.hdrfile.gridpar = nll.gridpar.grids.vp
    nll.init_control_file()

    return nll


class NLL(object):

    def __init__(self, project_code, base_folder='NLL', gridpar=None,
                 sensors=None, params=None):
        """
        :param project_code: the name of project, to be used for generating
        file names
        :type project_code: str
        :param event: and event containing picks and an origin with arrivals
        referring to the picks
        :type event: ~microquake.core.event.Event
        :param base_folder: the name of the NLL folder
        :type base_folder: str
        """
        self.project_code = project_code
        self.base_folder = base_folder

        self.ctrlfile = NLLControl()
        self.hdrfile = NLLHeader()

        self.gridpar = gridpar
        self.sensors = sensors
        self.params = params

        self.hdrfile.gridpar = self.gridpar.grids.vp
        self.init_control_file()


    @property
    def base_name(self):
        return '%s' % self.project_code

    def _make_base_folder(self):
        try:
            if not os.path.exists(self.base_folder):
                os.mkdir(self.base_folder)
            if not os.path.exists(os.path.join(self.base_folder, 'run')):
                os.mkdir(os.path.join(self.base_folder, 'run'))
            if not os.path.exists(os.path.join(self.base_folder, 'model')):
                os.mkdir(os.path.join(self.base_folder, 'model'))
            if not os.path.exists(os.path.join(self.base_folder, 'time')):
                os.mkdir(os.path.join(self.base_folder, 'time'))
            return True
        except:
            return False

    def _clean_outputs(self):
        try:
            for f in glob(os.path.join(self.base_folder, 'loc',
                                       self.base_name)):
                os.remove(f)
        except:
            pass

    def _prepare_project_folder(self):

        self.worker_folder = tempfile.mkdtemp(dir=self.base_folder).split('/')[-1]

        os.mkdir(os.path.join(self.base_folder, self.worker_folder, 'loc'))
        os.mkdir(os.path.join(self.base_folder, self.worker_folder, 'obs'))
        logger.debug('%s.%s: cwd=%s' % (__name__,'_prepare_project_folder',
                                        os.getcwd()))

    def _finishNLL(self):

        '''
        file = "%s/run/%s_%s.in" % (self.base_folder, self.base_name, self.worker_folder)
        print("_finishNLL: Don't remove tmp=%s/%s" % (self.base_folder, self.worker_folder))
        return
        '''

        os.remove('%s/run/%s_%s.in' % (self.base_folder, self.base_name,
                                       self.worker_folder))
        self._clean_outputs()
        tmp = '%s/%s' % (self.base_folder, self.worker_folder)
        shutil.rmtree(tmp)


    def init_header_file(self):
        """
        """
        pass

    def init_control_file(self):
        """
        """
        self.ctrlfile.vggrid = "VGGRID %s" % (str(self.hdrfile))

        if self.gridpar.homogeneous:
            laymod = "LAYER    %f  %f 0.00    %f  0.00  2.7 0.0" % (
                self.gridpar.grids.vp.origin[2] / 1000,
                self.gridpar.vp / 1000,
                self.gridpar.vs / 1000)

            modelname = self.project_code
        else:
            laymod = "LAYER"
            modelname = self.project_code

        modelname = '%s' % modelname

        self.ctrlfile.laymod = laymod
        self.ctrlfile.modelname = modelname
        self.ctrlfile.basefolder = self.base_folder

        # hdr = "%d %d %d  %.2f %.2f %.2f  %.4f %.4f %.4f  SLOW_LEN" % (
        self.ctrlfile.locgrid = "LOCGRID  %d %d %d  %.2f %.2f %.2f  %.4f %.4f %.4f  MISFIT  SAVE" % (
            (self.gridpar.grids.vp.shape[0] - 1) * 10 + 1,
            (self.gridpar.grids.vp.shape[1] - 1) * 10 + 1,
            (self.gridpar.grids.vp.shape[2] - 1) * 10 + 1,
            self.gridpar.grids.vp.origin[0] / 1000,
            self.gridpar.grids.vp.origin[1] / 1000,
            self.gridpar.grids.vp.origin[2] / 1000,
            self.gridpar.grids.vp.spacing / 10000,
            self.gridpar.grids.vp.spacing / 10000,
            self.gridpar.grids.vp.spacing / 10000)

        self.ctrlfile.locsig = self.params.locsig
        self.ctrlfile.loccom = self.params.loccom
        self.ctrlfile.locsearch = self.params.locsearch
        self.ctrlfile.locmeth = self.params.locmeth

        self.ctrlfile.phase = 'P'
        self.ctrlfile.vgtype = 'P'

        self.ctrlfile.basefolder = self.base_folder
        self.ctrlfile.projectcode = self.project_code

        try:
            self.ctrlfile.add_stations(self.sensors.name, self.sensors.pos)
        except:
            logger.error('Sensor file does not exist')

    def _write_velocity_grids(self):
        if not self.gridpar.homogeneous:
            if self.gridpar.vp:
                p_file = '%s/model/%s.P.mod' % (self.base_folder,
                        self.base_name)
                self.gridpar.grids.vp.write(p_file, format='NLLOC')
                self.gridpar.filep = self.gridpar.vs.split('/')[-1]
            else:
                self.gridpar.filep = None

            if self.gridpar.vs:
                s_file = '%s/model/%s.S.mod' % (self.base_folder,
                        self.base_name)
                self.gridpar.grids.vs.write(s_file, format='NLLOC')

                self.gridpar.files = self.gridpar.vs.split('/')[-1]
            else:
                self.gridpar.files = None

        if self.gridpar.homogeneous:
            self.ctrlfile.vgout = '%s/model/%s' % (self.base_folder,
                    self.base_name)
            self.ctrlfile.vgout = '%s/model/%s' % (self.base_folder,
                    self.base_name)

        else:
            self.ctrlfile.vgout = '%s/model/%s.P.mod.buf' % (self.base_folder,
                                                             self.base_name)
            self.ctrlfile.vgout = '%s/model/%s.S.mod.hdr' % (self.base_folder,
                                                             self.base_name)

    def prepare(self, create_time_grids=True, create_angle_grids=True,
                create_distance_grids=False, tar_files=False):
        """
        Creates the NLL folder and prepare the NLL configuration files based on the
        given configuration

        :param create_time_grids: if True, runs Vel2Grid and Grid2Time
        :type create_time_grids: bool
        :param tar_files: creates a tar of the NLL library
        :type tar_files: bool
        """

        logger.debug(os.getcwd())
        self._make_base_folder()
        logger.debug(os.getcwd())

        self.hdrfile.write('%s/run/%s.hdr' % (self.base_folder, self.base_name))
        self._write_velocity_grids()
        self.ctrlfile.write('%s/run/%s.in' % (self.base_folder, self.base_name))

        if create_time_grids:
            self._create_time_grids()

        if create_angle_grids:
            self._create_angle_grids()

        if create_distance_grids:
            self._create_distance_grids()

        if tar_files:
            self.tar_files()

    def _create_time_grids(self):
        self.ctrlfile.phase = 'P'
        self.ctrlfile.vgtype = 'P'
        self.ctrlfile.write('%s/run/%s.in' % (self.base_folder, self.base_name))
        if self.gridpar.vp:
            if self.gridpar.homogeneous:
                logger.debug('Creating P velocity grid')
                cmd = 'Vel2Grid %s/run/%s.in' % (self.base_folder,
                        self.base_name)
                os.system(cmd)

            logger.debug('Calculating P time grids')
            cmd = 'Grid2Time %s/run/%s.in' % (self.base_folder, self.base_name)
            os.system(cmd)

        if self.gridpar.vs:
            self.ctrlfile.phase = 'S'
            self.ctrlfile.vgtype = 'S'
            self.ctrlfile.write('%s/run/%s.in' % ( self.base_folder,
                self.base_name))
            if self.gridpar.homogeneous:
                logger.debug('Creating S velocity grid')
                cmd = 'Vel2Grid %s/run/%s.in' % (self.base_folder,
                        self.base_name)
                os.system(cmd)

            logger.debug('Calculating S time grids')
            cmd = 'Grid2Time %s/run/%s.in' % (self.base_folder, self.base_name)
            os.system(cmd)

    def _create_angle_grids(self):
        """
        calculate and write angle grids from travel time grids
        """

        from glob import glob
        time_files = glob('%s/time/*time*.hdr' % self.base_folder)

        for time_file in time_files:
            self._save_angle_grid(time_file)
        # map(self._save_angle_grid, time_files)

    def _save_angle_grid(self, time_file):
        """
        calculate and save take off angle grid
        """
        from microquake.simul.eik import angles
        from microquake.core import read_grid
        from IPython.core.debugger import Tracer
        # reading the travel time grid
        ifile = time_file
        ttg = read_grid(ifile, format='NLLOC')
        az, toa = angles(ttg)
        tmp = ifile.split('/')
        tmp[-1] = tmp[-1].replace('time', 'take_off')
        # Tracer()()
        ofname = '/'.join(tmp)
        toa.write(ofname, format='NLLOC')
        az.write(ofname.replace('take_off', 'azimuth'), format='NLLOC')

    def _create_distance_grids(self):
        """
        create distance grids using the ray tracer. Will take long time...
        Returns:

        """
        from glob import glob
        from microquake.core.data.grid import read_grid
        from numpy import arange, meshgrid, zeros_like
        from microquake.simul.eik import ray_tracer
        from time import time
        time_files = glob('%s/time/*time*.hdr' % self.base_folder)

        ttg = read_grid(time_files[0], format='NLLOC')
        x = arange(0, ttg.shape[0])
        y = arange(0, ttg.shape[1])
        z = arange(0, ttg.shape[2])

        X, Y, Z = meshgrid(x, y, z)
        X = X.reshape(np.product(ttg.shape))
        Y = Y.reshape(np.product(ttg.shape))
        Z = Z.reshape(np.product(ttg.shape))

        out_array = zeros_like(ttg.data)

        for time_file in time_files:
            ttg = read_grid(time_file, format='NLLOC')
            for coord in zip(X, Y, Z):
                st = time()
                ray = ray_tracer(ttg, coord, grid_coordinates=True,
                                 max_iter=100)
                et = time()
                #print(et - st)
                out_array[coord[0], coord[1], coord[2]] = ray.length()

            tmp = time_file.split('/')
            tmp[-1] = tmp[-1].replace('time', 'distance')
            ofname = '/'.join(tmp)

            ttg.type = 'DISTANCE'
            ttg.write(ofname, format='NLLOC')

            retur

    def tar_files(self):
        # Create tar.gz from the NLL folder
        script = """
        tar -czvf NLL.tar.gz %s/*
        """ % (self.base_folder)

        with open('tmp.sh', 'w') as shfile:
            shfile.write(script)

        logger.debug('Preparing NLL tar file...')
        os.system('sh tmp.sh')
        os.remove('tmp.sh')

    def run_event(self, event, silent=True):
        fname = 'run_event'

        from glob import glob

        evt = event

        self._prepare_project_folder()

        ### TODO
        # MTH: If input event has no preferred_origin(), gen_observations
        # will (incorrectly) create one!
        event2 = self.gen_observations_from_event(evt)

        new_in = '%s/run/%s_%s.in' % (self.base_folder, self.base_name, self.worker_folder)
        #print("new_in=%s" % new_in)

        self.ctrlfile.workerfolder = self.worker_folder
        self.ctrlfile.write(new_in)

        os.system('NLLoc %s' % new_in)

        filename = "%s/%s/loc/last.hyp" % (self.base_folder, self.worker_folder)
        logger.debug('%s.%s: scan hypo from filename = %s' % (__name__,fname,filename))

        if not glob(filename):
            logger.error("%s.%s: location failed" % (__name__,fname))
            return Catalog(events=[evt])

        if event.origins:
            if event.preferred_origin():
                logger.debug('%s.%s: event.pref_origin exists --> set eval mode' % (__name__,fname))
                evaluation_mode = event.preferred_origin().evaluation_mode
                evaluation_status = event.preferred_origin().evaluation_status
            else:
                logger.debug('%s.%s: event.pref_origin does NOT exist --> set eval mode on origins[0]' % (__name__,fname))
                evaluation_mode = event.origins[0].evaluation_mode
                evaluation_status = event.origins[0].evaluation_status

        cat_out = self.read_hyp_loc(filename, event=event,
                                    evaluation_mode=evaluation_mode,
                                    evaluation_status=evaluation_status)

        self._finishNLL()
        return cat_out

    def gen_observations_from_event(self, event):
        """
        Create NLLoc compatible observation file from an microquake event catalog file.
        input:

        :param event: event containing a preferred origin with arrivals
        referring to picks
        :type event: ~microquake.core.event.Event
        """

        fname = 'gen_observations_from_event'

        with open('%s/%s/obs/%s.obs' % (self.base_folder, self.worker_folder,
                                        self.base_name), 'w') as out_file:
            po = event.preferred_origin()
            logger.debug('%s.%s: pref origin=[%s]' % (__name__,fname,po))

            if not po:
                logger.error('preferred origin is not set')

            for arr in po.arrivals:

                pk = arr.pick_id.get_referred_object()
                #logger.debug(pk)
                if pk.evaluation_status == 'rejected':
                    continue

                date_str = pk.time.strftime('%Y%m%d %H%M %S.%f')

                if pk.phase_hint == 'P':
                    pick_error = '1.00e-03'
                else:
                    pick_error = '1.00e-03'

                polarity = 'U' if pk.polarity == 'positive' else 'D'

                out_file.write('%s ?    ?    ?    %s %s %s GAU %s -1.00e+00 -1.00e+00 -1.00e+00\n'
                               % (pk.waveform_id.station_code.ljust(6),
                                  pk.phase_hint.ljust(6), polarity, date_str,
                                  pick_error))
        return event

    def read_hyp_loc(self, hypfile, event, evaluation_mode='automatic',
                     evaluation_status='preliminary', use_ray_tracer=True):
        """
        read the hypocenter file generate by the location run
        :param hypfile: path to hypocenter file generated by the NLLoc location
        run
        :type hypfile: str
        :param event: an event object with picks
        :type event: microquake.core.Event.event
        :param evaluation_mode: evaluation mode
        :type evaluation_mode: str
        :param evaluation_status: evaluation status
        :type evaluation_status: str
        :param use_ray_tracer: if true use ray tracer to measure
        event-station distance (default: True)
        :type use_ray_tracer: bool
        :rtype: ~microquake.core.event.Catalog
        """
        from microquake.core import read_grid
        from microquake.simul.eik import ray_tracer
        from time import time

        origin = read_nlloc_hypocenter_file(hypfile, event.picks,
                                            evaluation_mode=evaluation_mode,
                                            evaluation_status=evaluation_status)

        logger.info('ray tracing')
        st = time()
        if use_ray_tracer:
            for arrival in origin.arrivals:
                sensor_id = arrival.get_pick().waveform_id.station_code
                phase = arrival.phase

                fname = '%s.%s.%s.time' % (self.base_name, phase, sensor_id)

                fpath = os.path.join(self.base_folder, 'time', fname)

                ttg = read_grid(fpath, format='NLLOC')
                ray = ray_tracer(ttg, origin.loc, grid_coordinates=False)

                '''
                dist = arrival.distance
                pk = arrival.pick_id.get_referred_object()
                sta = pk.waveform_id.station_code
                '''

                arrival.distance = ray.length()

                #print("nlloc read_hyp_loc: arr sta:%s pha:%s dist:%f ray_dist:%f" % \
                      #(sta, arrival.phase, dist, arrival.distance))

                # arrival.ray = ray.nodes
        et = time()
        logger.info('completed ray tracing in %0.3f' % (et - st))


        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id

        return Catalog(events=[event])

    def take_off_angle(self, station):
        from microquake.core.data.grid import read_grid
        fname = '%s/time/%s.P.%s.take_off' % (self.base_folder, self.base_name,
                                          station)
        return read_grid(fname, format='NLLOC')


class NLLHeader(AttribDict):

    attributes = ['gridpar']

    def __str__(self):
        gridpar = self.gridpar
        hdr = "%d %d %d  %.4f %.4f %.4f  %.4f %.4f %.4f  SLOW_LEN" % (
            gridpar.shape[0],
            gridpar.shape[1],
            gridpar.shape[2],
            gridpar.origin[0] / 1000.,
            gridpar.origin[1] / 1000.,
            gridpar.origin[2] / 1000.,
            gridpar.spacing / 1000.,
            gridpar.spacing / 1000.,
            gridpar.spacing / 1000.)
        # hdr = self.__hdr_tmpl.replace(token,hdr)
        return hdr

    def __init__(self, *args, **kwargs):
        super(NLLHeader, self).__init__(*args, **kwargs)
        for attr in self.attributes:
            self[attr] = ''

    def read(self, fname):
        with open(fname, 'r') as fin:
            line = fin.readline()
            line = line.split()
            self.gridpar = AttribDict()
            self.gridpar.grids = AttribDict()
            self.gridpar.grids.v = AttribDict()
            self.gridpar.shape = tuple([int(line[0]), int(line[1]), int(line[2])])
            self.gridpar.origin = np.array([float(line[3]), float(line[4]), float(line[5])])
            self.gridpar.origin *= 1000
            self.gridpar.spacing = float(line[6]) * 1000

    def write(self, fname):
        with open(fname, 'w') as fout:
            token = '<HDR>'
            hdr = self.__hdr_tmpl.replace(token, str(self))
            fout.write(hdr)

    __hdr_tmpl = \
        """<HDR>
TRANSFORM  NONE
"""

supported_nlloc_grid_type = ['VELOCITY', 'VELOCITY_METERS', 'SLOWNESS',
                             'SLOW_LEN', 'TIME', 'PROB_DENSITY', 'MISFIT',
                             'ANGLE',]


valid_nlloc_grid_type = ['VELOCITY', 'VELOCITY_METERS', 'SLOWNESS', 'VEL2',
                         'SLOW2', 'SLOW2_METERS', 'SLOW_LEN', 'TIME', 'TIME2D',
                         'PROB_DENSITY', 'MISFIT', 'ANGLE', 'ANGLE2D']



class NLLControl(AttribDict):
    """
    NLLoc control file builder
    """

    tokens = ['workerfolder', 'projectcode', 'basefolder', 'modelname',
              'vgout', 'vgtype', 'vggrid', 'laymod',
              'loccom', 'locsig', 'locsearch',
              'locgrid', 'locmeth', 'modelname',
              'phase', 'gtsrce']

    def __str__(self):
        ctrl = self.__ctrl_tmpl
        for attr in self.tokens:
            token = '<%s>' % attr.upper()
            ctrl = ctrl.replace(token, self.__dict__[attr])
        return ctrl

    def __init__(self, *args, **kwargs):
        super(NLLControl, self).__init__(*args, **kwargs)
        for attr in self.tokens:
            self[attr] = ''

    def add_stations(self, sname, sloc):

        for n, l in zip(sname, sloc):
            l2 = l / 1000
            if len(n) > 6:
                logger.critical('NLL cannot handle station names longer than'
                                ' 6 characters, Sensor %s currently has %d'
                                ' characters' %(n, len(n)))
                logger.warning('Sensor %s will not be processed' % n)
                continue
            # noinspection PyStringFormat
            self.gtsrce += 'GTSRCE %s XYZ %f %f %f 0.00\n' % ((n,) + tuple(l2))

    def write(self, fname):
        with open(fname, 'w') as fout:
            fout.write(str(self))

    __ctrl_tmpl = \
"""
CONTROL 0 54321
TRANS NONE
VGOUT  <VGOUT> #<BASEFOLDER>/model/layer

VGTYPE P
VGTYPE S

<VGGRID>

<LAYMOD>

GTFILES  <BASEFOLDER>/model/<MODELNAME>  <BASEFOLDER>/time/<MODELNAME> <PHASE>

GTMODE GRID3D ANGLES_NO
# MTH Uncomment these if you want Grid2Time to calculate angles.buf (takeoff + azimuth)
#     and for the resulting angles to appear on the last.hyp phase lines
#GTMODE GRID3D ANGLES_YES
#LOCANGLES ANGLES_YES 5

<GTSRCE>

GT_PLFD  1.0e-3  0

LOCSIG Microquake package

LOCCOM created automatically by the microquake package

LOCFILES <BASEFOLDER>/<WORKERFOLDER>/obs/<MODELNAME>.obs NLLOC_OBS <BASEFOLDER>/time/<MODELNAME>  <BASEFOLDER>/<WORKERFOLDER>/loc/<MODELNAME>

#LOCHYPOUT SAVE_NLLOC_ALL SAVE_HYPOINV_SUM SAVE_NLLOC_OCTREE
LOCHYPOUT SAVE_NLLOC_ALL

LOCSEARCH <LOCSEARCH>

<LOCGRID>

LOCMETH <LOCMETH>

LOCGAU 0.001 0

LOCGAU2 0.001 0.001 0.001

LOCPHASEID  P   P p
LOCPHASEID  S   S s

LOCQUAL2ERR 0.0001 0.0001 0.0001 0.0001 0.0001

LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9
"""
