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
    from glob import glob
    cat = event.Catalog()

    with open(filename) as hyp_file:

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
            # origin.evaluation_status = 'rejected'
            evaluation_status = 'rejected'
            logger.warning('Event located on grid boundary')
        else:
            evaluation_status = evaluation_status
            # origin.evaluation_status = evaluation_status

        hyp_x = float(hyp[2]) * 1000
        hyp_y = float(hyp[4]) * 1000
        hyp_z = float(hyp[6]) * 1000

        # adding a new origin
        #method = '%s %s' % (sign[3], search[1])
        method = '%s %s' % ("NLLOC", search[1])

        # origin.x = hyp_x
        # origin.y = hyp_y
        # origin.z = hyp_z
        # origin.time = tme
        # origin.evaluation_mode = evaluation_mode # why automatic here
        # origin.epicenter_fixed = int(0)
        # origin.method = method
        # creation_info = event.CreationInfo()
        creation_info = event.CreationInfo(author='microquake',
                                           creation_time=UTCDateTime.now())

        # origin.creation_info.author = 'microquake'
        # origin.creation_info.creation_time = UTCDateTime.now()

        origin = event.Origin(x=hyp_x, y=hyp_y, z=hyp_z, time=tme,
                              evaluation_mode=evaluation_mode,
                              evaluation_status=evaluation_status,
                              epicenter_fixed=0, method=method,
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
            major_az = -9. 
        if np.isnan(major_dip):
            major_dip = -9. 

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
                sdist = float(tmp[21])
                azi = float(tmp[23])
                toa = float(tmp[24])

                dist = np.linalg.norm([sx * 1000 - origin.x,
                                       sy * 1000 - origin.y,
                                       sz * 1000 - origin.z])

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
        # phases = np.array(phases)

        points = read_scatter_file(filename.replace('.hyp','.scat'))

        origin.arrivals = [arr for arr in arrivals]
        origin.scatter = str(points)

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
        :param project: the name of the project
        :param perturbation:
        :return: microquake.core.event.Event
        """

        from microquake.core.data.grid import read_grid
        import numpy as np

        narr = len(event.preferred_origin().arrivals)

        # initializing the frechet derivative
        Frechet = np.zeros([narr, 3])

        event_loc = np.array(event.preferred_origin().loc)

        for i, arrival in enumerate(event.preferred_origin().arrivals):
            pick = arrival.pick_id.get_referred_object()
            station_code = pick.waveform_id.station_code
            phase = arrival.phase

            # loading the traveltime grid
            filename = '%s/time/%s.%s.%s.time' % (base_directory,
            base_name, phase, station_code)

            tt = read_grid(filename, format='NLLOC')
            spc = tt.spacing

            #calculate the frechet derivative
            for dim in range(0,3):
                loc_p1 = event_loc.copy()
                loc_p2 = event_loc.copy()
                loc_p1[dim] += perturbation
                loc_p2[dim] -= perturbation
                tt_p1 = tt.interpolate(loc_p1, grid_coordinate=False)
                tt_p2 = tt.interpolate(loc_p2, grid_coordinate=False)
                Frechet[i, dim] = (tt_p1 - tt_p2) / (2 * perturbation)

        hessian = np.linalg.inv(np.dot(Frechet.T, Frechet))
        tmp = hessian * pick_uncertainty ** 2
        w, v = np.linalg.eig(tmp)
        return w, v


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
    # Testing the presence of the .buf or .hdr extension at the end of base_name
    if ('.buf' == base_name[-4:]) or ('.hdr' == base_name[-4:]):
        # removing the extension
        base_name = base_name[:-4]

    # Reading header file
    try:
        head = _read_nll_header_file(base_name + '.hdr')
    except:
        logger.error('error reading %s' % base_name + '.hdr')

    # Read binary buffer
    gdata = np.fromfile(base_name + '.buf', dtype=np.float32)
    gdata = gdata.reshape(head.shape)
    if head.grid_type == 'SLOW_LEN':
        gdata = head.spacing / gdata
        head.grid_type = 'VELOCITY'
        
    return GridData(gdata, spacing=head.spacing, origin=head.origin,
            seed=head.seed, seed_label=head.label, grid_type=head.grid_type)


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


def write_nll_grid(base_name, data, origin, spacing, grid_type,
        seed=None, label=None, velocity_to_slow_len=True):
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
        self.project_folder = os.getcwd()
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

        self.project_folder = os.getcwd()
        self.worker_folder = tempfile.mkdtemp(dir=self.base_folder).split('/')[-1]

        os.mkdir(os.path.join(self.base_folder, self.worker_folder, 'loc'))
        os.mkdir(os.path.join(self.base_folder, self.worker_folder, 'obs'))
        logger.debug('%s.%s: cwd=%s' % (__name__,'_prepare_project_folder',
                                        os.getcwd()))

    def _finishNLL(self):

        '''
        file = "%s/run/%s_%s.in" % (self.base_folder, self.base_name, self.worker_folder)
        print("_finishNLL: Don't remove:%s" % file)
        print("_finishNLL: Don't remove tmp=%s/%s" % (self.base_folder, self.worker_folder))
        '''
        os.remove('%s/run/%s_%s.in' % (self.base_folder, self.base_name,
                                       self.worker_folder))
        self._clean_outputs()
        tmp = '%s/%s' % (self.base_folder, self.worker_folder)
        shutil.rmtree(tmp)
        # os.chdir(self.project_folder)

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

    def prepare(self, create_time_grids=True, tar_files=False):
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
        self.project_folder = os.getcwd()
        logger.debug(os.getcwd())

        self.hdrfile.write('%s/run/%s.hdr' % (self.base_folder, self.base_name))
        self._write_velocity_grids()
        self.ctrlfile.write('%s/run/%s.in' % (self.base_folder, self.base_name))

        if create_time_grids:
            self._create_time_grids()

        if tar_files:
            self.tar_files()

    def _save_time_grid(self, station, velocity, phase='P'):
        """
        calculate and save a travel time grid in the <time> directory
        :param station: a station to use as a seed to calculate the travel time
        grid
        :type station: ~microquake.core.data.station.Station
        :param velocity: velocity grid
        :type velocity: ~microquake.core.data.grid.GridData
        :param phase: the phase associated with the velocity 'P' or 'S' (not
        case sensitive)
        :type phase: str
        """

        from microquake.simul import eik
        stloc = station.loc
        phase = phase.upper()
        tt_grid = eik.eikonal_solver(velocity, stloc)
        tt_grid.write('%s/time/%s.%s.%s.time' % (self.base_folder,
            self.base_name, phase, station.code), format='NLLOC')

        # calculating the azimuth and take off angles
        az, toa = eik.angles(tt_grid)
        toa.write('%s/time/%s.%s.%s.angle' % (self.base_folder,
            self.base_name, phase, station.code), format='NLLOC')
        return

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
            #print('MTH: microquake nlloc create_time_grids: cmd = [%s]' % cmd)
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

            # overriding angle grids
            # the angle grids generated by NLLoc need to be overriden as angle
            # are always 0
            from glob import glob
            time_files = glob('%s/time/*time*.hdr' % self.base_folder)
            self._create_angle_grid(time_files)

    def _create_angle_grid(self, time_files):
        """
        calculate and write angle grids from travel time grids
        :param time_files: list of files containing travel time information
        :type time_files: iterable of travel time file paths
        :param SparkContext: spark context to run operation in parallel
        :type SparkContext: pyspark.SparkContext
        """
        map(self._save_angle_grid, time_files)

    def _save_angle_grid(self, time_file):
        """
        calculate and save take off angle grid
        """
        from microquake.simul.eik import angles
        from microquake.core import read_grid
        # reading the travel time grid
        ifile = time_file
        ttg = read_grid(ifile, format='NLLOC')
        az, toa = angles(ttg)
        tmp = ifile.split('/')
        tmp[-1] = tmp[-1].replace('time', 'take_off')
        ofname = '/'.join(tmp)
        toa.write(ofname, format='NLLOC')
        az.write(ofname.replace('take_off', 'azimuth'), format='NLLOC')

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
        self.ctrlfile.workerfolder = self.worker_folder
        self.ctrlfile.write(new_in)

        os.system('NLLoc %s' % new_in)

        filename = "%s/%s/loc/last.hyp" % (self.base_folder, self.worker_folder)
        logger.debug('%s.%s: scan hypo from filename = %s' % (__name__,fname,filename))

        if not glob(filename):
            #self._finishNLL()
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

    def read_hyp_loc(self, hypfile, event=None, evaluation_mode='automatic',
                     evaluation_status='preliminary'):
        """
        read the hypocenter file generate by the location run
        :param hypfile: path to hypocenter file generated by the NLLoc location
        run
        :type hypfile: str
        :rtype: ~microquake.core.event.Catalog
        """
        from microquake.core import read_grid
        from numpy import pi

        origin = read_nlloc_hypocenter_file(hypfile, event.picks,
                                            evaluation_mode=evaluation_mode,
                                            evaluation_status=evaluation_status)

        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id

        return Catalog(events=[event])

    def take_off_angle(self, station):
        from microquake.core.data.grid import read_grid
        fname = '%s/time/%s.P.%s.time' % (self.base_folder, self.base_name,
                                          station)
        gd = read_grid(fname, format='NLLOC')


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


def gdef_to_points(gdef):
    
    shape, origin, spacing = gdef[:3], gdef[3:6], float(gdef[6])
    # nx, ny, nz = shape
    maxes = origin + shape * spacing
    x = np.arange(origin[0], maxes[0], spacing).astype(np.float32)
    y = np.arange(origin[1], maxes[1], spacing).astype(np.float32)
    z = np.arange(origin[2], maxes[2], spacing).astype(np.float32)
    points = np.zeros((np.product(shape), 3), dtype=np.float32)
    # points = np.stack(np.meshgrid(x, y, z), 3).reshape(3, -1).astype(np.float32)
    ix = 0
    for xv in x:
        for yv in y:
            for zv in z:
                points[ix] = [xv, yv, zv]
                ix += 1
    return points


def read_nll_header(fle):
    # print(fle)
    dat = open(fle).read().split()
    shape = np.array(dat[:3], dtype=int)
    org = np.array(dat[3:6], dtype=np.float32) * 1000.
    spacing = (np.array(dat[6:9], dtype=np.float32) * 1000.)[0]
    sloc = np.array(dat[12:15], dtype=np.float32) * 1000.

    return sloc, shape, org, spacing


def ttable_from_nll_grids(path, key="OT.P"):
    fles = np.sort(glob(os.path.join(path, key + '*.time.buf')))
    hfles = np.sort(glob(os.path.join(path, key + '*.time.hdr')))
    assert(len(fles) == len(hfles))
    stas = np.array([f.split('.')[-3].zfill(3) for f in fles], dtype='S4')
    isort = np.argsort(stas)
    fles = fles[isort]
    hfles = hfles[isort]
    names = stas[isort]

    vals = [read_nll_header(fle) for fle in hfles]
    sloc, shape, org, spacing = vals[0]
    slocs = np.array([v[0] for v in vals], dtype=np.float32)
    ngrid = np.product(shape)

    nsta = len(fles)
    tts = np.zeros((nsta, ngrid), dtype=np.float32)

    for i in range(nsta):
        tts[i] = np.fromfile(fles[i], dtype='f4')

    gdef = np.concatenate((shape, org, [spacing])).astype(np.int32)

    ndict = {}
    for i, sk in enumerate(names):
        ndict[sk.decode('utf-8')] = i

    return tts, slocs, ndict, gdef
