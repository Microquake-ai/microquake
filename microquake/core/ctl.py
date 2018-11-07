# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
#  Purpose: module to read control files
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
module to interact with and read control files

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import itertools

import numpy as np

# import pickle as pickle
from xml.dom import minidom
from microquake.core.util.attribdict import AttribDict
from microquake.core.data import grid
from microquake.core import logger
from glob import glob


def parse_one(el, tag):
    return parse_many(el, tag)[0]


def parse_many(el, tag):
    return el.getElementsByTagName(tag)


def parse_attrib(el, a, cast=None):
    if isinstance(a, list):
        return np.array([parse_attrib(el, x, cast) for x in a])
    else:
        if cast is None:
            return el.getAttribute(a)
        else:
            return cast(el.getAttribute(a))


def parse_one_val(el, tag, cast=None):
    el2 = parse_one(el, tag)
    return parse_attrib(el2, "value", cast=cast)


def parse_project(ctl, params):

    project_el = parse_one(ctl, "project")
    db = parse_one(project_el, "db")
    format_continuous = parse_one(project_el, "format_continuous")
    waveforms = parse_one(project_el, "waveforms")
    events = parse_one(project_el, "events")
    template = parse_one(project_el, "template")

    project = AttribDict()
    project.db_type, project.db_uri, project.db_name = parse_attrib(db, ["type", "uri", "db_name"])
    project.waveform_path = parse_attrib(waveforms, "path")
    project.event_path = parse_attrib(events, "path")
    project.template_dir, project.template_file = parse_attrib(template, ["directory", "file"])
    project.format_continuous = parse_attrib(format_continuous, "value")
    params.project = project


def parse_file_loc(ctl, params):

    try:
        file_location = parse_one(ctl, "file_locations")
        common_file = parse_attrib(parse_one(file_location, "common_file_location"), "value")

        # environment variable
        if "$" in common_file:
            import os
            params.common_file_location = os.environ[common_file[1:]]
        else:
            params.common_file_location = common_file
    except:
        params.common_file_location = './'


def parse_grid(ctl, params):

    # Prepare grid
    grid_el = parse_one(ctl, "grid")
    defin = parse_one(grid_el, "definition")
    params.grid_method = parse_attrib(defin, "method")

    if params.grid_method == "ODS":
        grid_units = parse_one(defin, "units")
        origin = parse_one(defin, "origin")
        spacing = parse_one(defin, "spacing")
        dimensions = parse_one(defin, "dimensions")
        params.grid_units = parse_attrib(grid_units, "value")
        params.origin = parse_attrib(origin, ["X", "Y", "Z"], float)
        params.spacing = parse_attrib(spacing, "value", float)
        params.dimensions = parse_attrib(dimensions, ["nx", "ny", "nz"], int)

    elif params.grid_method == "OCS":
        grid_units = parse_one(defin, "units")
        origin = parse_one(defin, "origin")
        corner = parse_one(defin, "corner")
        spacing = parse_one(defin, "spacing")
        params.grid_units = parse_attrib(grid_units, "value")
        params.origin = parse_attrib(origin, ["X", "Y", "Z"], float)
        params.corner = parse_attrib(corner, ["X", "Y", "Z"], float)
        params.spacing = parse_attrib(spacing, "value", float)

    params.velgrids = parse_velgrid(ctl, params, grid_el)

    try:
        params.densitygrid = parse_densitygrid(ctl, params, grid_el)
    except:
        params.densitygrid = []

    try:
        params.attgrid = parse_attenuationgrid(ctl, params, grid_el)
    except:
        params.attgrid = []


    try:
        params.noisegrid = parse_noise_grid(ctl, params, grid_el)
    except:
        params.noisegrid = []


def parse_velgrid(ctl, params, grid_el):
    """
    parse velocity grid
    """
    velmodel = parse_one(grid_el, "velmodel")
    id = parse_attrib(velmodel, "id", int)
    velmethod = parse_one(velmodel, "method")
    homogeneous = parse_attrib(velmethod, "homogeneous")
    format = parse_attrib(velmodel, "format")
    homogeneous = True if homogeneous.lower() == "true" else False
    if homogeneous:
        vel = parse_one(velmodel, "velocity")
        griddict = AttribDict()
        grids = {}
        for t in ["vp", "vs"]:
            v = parse_attrib(vel, t, float)
            griddict[t] = v
            if params.grid_method == "ODS":
                grids[t] = grid.create('ODS',
                                       origin=params.origin,
                                       dimensions=params.dimensions,
                                       spacing=params.spacing,
                                       val=v)
            else:
                grids[t] = grid.create('OCS',
                                       origin=params.origin,
                                       corner=params.corner,
                                       spacing=params.spacing,
                                       val=v)

        griddict.homogeneous = homogeneous
        griddict.key = "%f" % griddict.vp
        griddict.grids = grids
        griddict.file = 'layer'

    else:
        vel = parse_one(velmodel, "velocity")
        format = parse_attrib(vel, "format").upper()
        griddict = AttribDict()

        grids = {}
        for t in ["vs", "vp"]:
            try:
                v = parse_attrib(vel, t)
            except:
                v = None
            griddict[t] = v
            from microquake.core.data.grid import read_grid
            import os
            vfile = os.path.join(params.common_file_location, v)
            grids[t] = read_grid(vfile, format=format)

        griddict.homogeneous = homogeneous
        # try:
        #     griddict.key = griddict.vs.split('/')[-1]
        # except:
        #     griddict.key = griddict.vp.split('/')[-1]

        griddict.grids = grids

    griddict.index = id
    return griddict


def parse_densitygrid(ctl, params, grid_el):
    denmodel = parse_one(grid_el, "denmodel")
    id = parse_attrib(denmodel, "id", int)
    denmethod = parse_one(denmodel, "method")
    homogeneous = parse_attrib(denmethod, "homogeneous")
    homogeneous = True if homogeneous.lower() == "true" else False
    if homogeneous:
        den = parse_one(denmodel, "density")
        r = parse_attrib(den, "rho", float)

        griddict = AttribDict()
        griddict.r = r
        griddict.homogeneous = homogeneous
        griddict.key = "%f" % r

        if params.grid_method == "ODS":
            Grid = grid.create('ODS',
                               origin=params.origin,
                               dimensions=params.dimensions,
                               spacing=params.spacing,
                               val=griddict.r)
        else:
            Grid = grid.create('OCS', origin=params.origin,
                               corner=params.corner,
                               spacing=params.spacing, val=griddict.r)

        griddict.data = Grid
    
    else:
        den = parse_one(denmodel, "density")
        griddict = AttribDict()

        griddict.file = parse_attrib(den, "file")
        griddict.units = parse_attrib(den, "units")

        griddict.homogeneous = homogeneous
        griddict.key = griddict.file.split('/')[-1]

        Grid = np.load(den)
        griddict.data = Grid

    griddict.index = id

    return griddict


def parse_attenuationgrid(ctl, params, grid_el):
    
    attmodel = parse_one(grid_el, "attmodel")
    id = parse_attrib(attmodel, "id", int)
    velmethod = parse_one(attmodel, "method")
    method = parse_attrib(velmethod, "homogeneous")
    method = True if method.lower() == "true" else False
    if method:
        att = parse_one(attmodel, "attenuation")

        griddict = AttribDict()
        grids = {}
        for t in ["qp", "qs"]:
            q = parse_attrib(att, t, float)
            griddict[t] = q

        
            if params.grid_method == "ODS":
                grids[t] = grid.create('ODS',
                                   origin=params.origin,
                                   dimensions=params.dimensions,
                                   spacing=params.spacing,
                                   val=q)
            else:
                grids[t] = grid.create('OCS', origin=params.origin,
                                   corner=params.corner,
                                   spacing=params.spacing, 
                                   val=q)

        griddict.homogeneous = method
        griddict.key = "%f" % griddict.qp
        griddict.grids = grids
        griddict.file = 'layer'

        # logger.debug('Saving Grid on disk')
        # pickle.dump(Grid, open("%s_grid.pickle" % (t), 'w'))

    else:
        att = parse_one(attmodel, "attenuation")
        griddict = AttribDict()

        grids = {}
        for t in ["qs", "qp"]:
            try:
                q = parse_attrib(att, t)
            except:
                q = None
            griddict[t] = q

            grids[t] = np.load(q)

        griddict.units = parse_attrib(att, "units")

        griddict.homogeneous = method

        try:
            griddict.key = griddict.qs.split('/')[-1]
        except:
            griddict.key = griddict.qp.split('/')[-1]


        griddict.grids = grids

    griddict.index = id

    return griddict


def parse_noise_grid(ctl, params, grid_el):
    noisemodel = parse_one(grid_el, "noisemodel")
    id = parse_attrib(noisemodel, "id", int)
    velmethod = parse_one(noisemodel, "method")
    method = parse_attrib(velmethod, "homogeneous")
    method = True if method.lower() == "true" else False
    if method:
        att = parse_one(noisemodel, "noise")

        griddict = AttribDict()
        grids = {}
        for t in ["qp", "qs"]:
            q = parse_attrib(att, t, float)
            griddict[t] = q


            if params.grid_method == "ODS":
                grids[t] = grid.create('ODS',
                                   origin=params.origin,
                                   dimensions=params.dimensions,
                                   spacing=params.spacing,
                                   val=q)
            else:
                grids[t] = grid.create('OCS', origin=params.origin,
                                   corner=params.corner,
                                   spacing=params.spacing,
                                   val=q)

        griddict.homogeneous = method
        griddict.key = "%f" % griddict.qp
        griddict.grids = grids
        griddict.file = 'layer'

        # logger.debug('Saving Grid on disk')
        # pickle.dump(Grid, open("%s_grid.pickle" % (t), 'w'))

    else:
        att = parse_one(noisemodel, "noise")
        griddict = AttribDict()

        grids = {}
        for t in ["noise"]:
            try:
                q = parse_attrib(att, t)
            except:
                q = None
            griddict[t] = q

            grids[t] = np.load(q)

        griddict.units = parse_attrib(att, "units")

        griddict.homogeneous = method


        griddict.grids = grids

    griddict.index = id

    return griddict


def parse_sensors(ctl, params):

    from microquake.core.data.station import read_stations
    import os

    params.sensors = []
    sens = parse_one(ctl, "sensors")
    id = parse_attrib(sens, "id", int)
    sfile = parse_one(sens, "file")
    sformat = parse_attrib(sfile, "format")
    spath = parse_attrib(sfile, "path")

    fname = os.path.join(params.common_file_location, spath)

    try:
        site = read_stations(fname, format=sformat, site_code=params.site)
    except:
        site = read_stations(fname, format=sformat, site_code=params.site,
                             has_header=True)

    snames = []
    spos = []
    for sta in site.stations():
        snames.append(sta.code)
        spos.append(sta.loc)

    sensdict = AttribDict()
    sensdict.key = "%d" % id
    sensdict.site = site
    sensdict.name = np.array(snames)
    sensdict.pos = np.array(spos)
    sensdict.index = id
    params.sensors = sensdict


def parse_events(ctl, params):

    Grid = params.velgrids.grids['vs']
    evparams = parse_one(ctl, "events")
    grd = []
    method = None
    ev_spacing = None
    eformat = None
    tmpgrid = None
    for j, event in enumerate(parse_many(evparams, "position")):
        method = parse_attrib(event, "method")
        if method == "ongrid":
            ev_spacing = parse_attrib(event, "spacing", float)

        if method == "file":
            eformat = parse_attrib(event, "format")
            efile = parse_attrib(event, "path")
            if eformat == "csv":
                tmpgrid = np.loadtxt(efile, skiprows=1, usecols=[0, 1, 2],
                                     delimiter=',')
                grd.append(tmpgrid)

    if method == "ongrid":
        evdata = AttribDict()
        evdata.data = np.array(grid.GenEventsOnGrid(Grid, ev_spacing))
        evdata.method = method
        evdata.eventSpacing = ev_spacing
        params.events = evdata

    if method == "file":
        if eformat == "csv":
            evdata = AttribDict()
            grd = np.array(grd).ravel()
            dim = tmpgrid.shape[1]
            grd = grd.reshape((len(grd) / dim, dim))
            evdata.data = grd
            evdata.method = method
            evdata.eventSpacing = 0
            params.events = evdata

    # need to implement a capability to read event location from csv or q64
    # files.


def parse_pick(ctl, params):

    params.pick_perturb = []
    pickparams = parse_one(ctl, "pickparams")
    for j, pick in enumerate(parse_many(pickparams, "perturbation")):
        error = parse_attrib(pick, "error", float)
        nsample = parse_attrib(pick, "nsample", int)
        for i in range(nsample):
            pickdict = AttribDict()
            pickdict.key = "%f_%d" % (error, i)
            pickdict.data = error
            pickdict.index = len(params.pick_perturb)
            params.pick_perturb.append(pickdict)


def parse_ancc_params(ctl, params):
    params.ancc = AttribDict()
    ancc_params = parse_one(ctl, "ancc")
    for node in ancc_params.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        try:
            params.ancc[node.tagName] = eval(node.getAttribute('value'))
        except:
            params.ancc[node.tagName] = node.getAttribute('value')


def parse_picker_params(ctl, params):
    params.picker = AttribDict()
    auto_picker_params = parse_one(ctl, "autopicker")
    for node in auto_picker_params.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue

        if (node.tagName == "STALTA_picker") or (node.tagName == "SNR_picker"):
            params.picker[node.tagName] = AttribDict()
            for prop in node.childNodes:

                if prop.nodeType != node.ELEMENT_NODE:
                    continue

                params.picker[node.tagName][prop.tagName] = AttribDict()
                for key in prop.attributes.keys():

                    params.picker[node.tagName][prop.tagName][key] = \
                        eval(prop.getAttribute(key))
        else:
            params.picker[node.tagName] = AttribDict()
            if node.hasAttribute('value'):
                params.picker[node.tagName] = eval(node.getAttribute('value'))
            else:
                params.picker[node.tagName] = AttribDict()
                for key in node.attributes.keys():
                    params.picker[node.tagName][key] = \
                        eval(node.getAttribute(key))


def parse_inversion(ctl, params):
    params.inversion = AttribDict()
    inversion_params = parse_one(ctl, 'inversion')
    for node in inversion_params.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        params.inversion[node.tagName] = eval(node.getAttribute('value'))


def parse_trigger_params(ctl, params):
    params.trigger = AttribDict()
    trigger_params = parse_one(ctl, 'trigger')
    for node in trigger_params.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        params.trigger[node.tagName] = eval(node.getAttribute('value'))


def parse_io_params(ctl, params):
    params.io = AttribDict()
    io_params = parse_one(ctl, 'io')
    for node in io_params.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        try:
            params.io[node.tagName] = eval(node.getAttribute('value'))
        except:
            params.io[node.tagName] = node.getAttribute('value')


def parse_db_params(ctl, params):
    params.db = AttribDict()
    db_params = parse_one(ctl, 'io')
    for node in db_params.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        try:
            params.db[node.tagName] = eval(node.getAttribute('value'))
        except:
            params.db[node.tagName] = node.getAttribute('value')


def parse_simulation(ctl, params):
    params.simulation = AttribDict()
    simulation_params = parse_one(ctl, 'simulation')
    for node in simulation_params.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == 'amplitude_unit_acceleration':
            value = node.getAttribute('value')
            params.simulation[node.tagName] = True if value.lower() == "true" else False
        else:
            try:
                params.simulation[node.tagName] = eval(node.getAttribute('value'))
            except:
                params.simulation[node.tagName] = node.getAttribute('value')


def parse_calc_source_params(ctl, params):
    params.calc_source = AttribDict()
    calc_source_params = parse_one(ctl, 'calcsourceparams')
    for node in calc_source_params.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        params.calc_source[node.tagName] = eval(node.getAttribute('value'))


def parse_calc_params(ctl, params):

    params.mag = []
    calcparams = parse_one(ctl, "calcparams")
    for node in calcparams.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue

        if node.tagName == "mag":
            mag = parse_attrib(node, "value", float)
            magdict = AttribDict()
            magdict.key = "%0.2f" % mag
            magdict.data = mag
            magdict.index = len(params.mag)
            params.mag.append(magdict)
        elif node.tagName == "NLL":
            import os
            nlldict = AttribDict()
            tmp = parse_one_val(node, "NLL_BASE", str)
            nlldict.NLL_BASE = os.path.join(params.common_file_location, tmp)
            nlldict.locsig = parse_one_val(node, "locsig", str)
            nlldict.loccom = parse_one_val(node, "loccom", str)
            nlldict.locsearch = parse_one_val(node, "locsearch", str)
            nlldict.locmeth = parse_one_val(node, "locmeth", str)
            params.nll = nlldict

        else:
            params[node.tagName] = eval(parse_attrib(node, "value"))


# def parse_spark_params(ctl, params):
#     params.spark = AttribDict()
#     sparkparams = parse_one(ctl, "spark")
#     for node in sparkparams.childNodes:
#         if node.nodeType != node.ELEMENT_NODE:
#             continue
#         params.spark[node.tagName] = node.getAttribute('value')

#     try:
#         from pyspark import SparkConf
#         conf = SparkConf()
#         for node in sparkparams.childNodes:
#             if node.nodeType != node.ELEMENT_NODE:
#                 continue
#             conf.set(node.tagName,  node.getAttribute('value'))

#         params.spark.conf = conf

#     except Exception as e:
#         params.spark.conf = None
#         logger.warning('Problem with pyspark')

    
def parse_control_file(filename, section=None):

    ctl = minidom.parse(filename)
    root = parse_one(ctl, "microquake")
    filetype = parse_attrib(root, "filetype")
    spark_enabled = parse_attrib(root, "spark_enabled")
    spark_enabled = True if spark_enabled.lower() == "true" else False

    if not filetype:
        filetype = 'systemdesign'

    params = AttribDict()
    project_code = parse_attrib(root, "project_code")

    site = parse_attrib(root, "site")
    if not project_code:
        project_code = 'microquake'
    if not site:
        site = "microquake"

    if spark_enabled:
        parse_spark_params(ctl, params)

    params.project_code = project_code
    params.spark_enabled = spark_enabled
    params.site = site

    parse_file_loc(ctl, params)

    if section is not None:
        if section == 'grd':
            parse_grid(ctl, params)
        elif section == 'sensors':
            parse_sensors(ctl, params)
        elif section == 'events':
            parse_events(ctl, params)
        elif section == 'pick':
            parse_pick(ctl, params)
        elif section == 'calc':
            parse_calc_params(ctl, params)
        elif section == 'picker':
            parse_picker_params(ctl, params)
        elif section == 'trigger':
            parse_trigger_params(ctl, params)
        elif section == 'calc_source':
            parse_calc_source_params(ctl, params)
        elif section == 'project':
            parse_project(ctl, params)

        return params

    if filetype == 'system-design':
        parse_grid(ctl, params)
        parse_sensors(ctl, params)
        parse_events(ctl, params)
        parse_simulation(ctl, params)

    elif filetype == 'automatic-processing':
        parse_grid(ctl, params)
        parse_sensors(ctl, params)
        parse_calc_params(ctl, params)
        parse_trigger_params(ctl, params)
        parse_picker_params(ctl, params)
        parse_calc_source_params(ctl, params)

    elif filetype == 'production-automatic-processing':
        parse_grid(ctl, params)
        parse_sensors(ctl, params)
        parse_calc_params(ctl, params)
        parse_trigger_params(ctl, params)
        parse_picker_params(ctl, params)
        parse_calc_source_params(ctl, params)
        parse_io_params(ctl, params)
        parse_db_params(ctl, params)

    elif filetype == 'project':
        parse_project(ctl, params)
        parse_grid(ctl, params)
        parse_sensors(ctl, params)
        parse_calc_params(ctl, params)
        parse_trigger_params(ctl, params)
        parse_picker_params(ctl, params)
        parse_calc_source_params(ctl, params)

    elif filetype == "ambient-noise":
        parse_grid(ctl, params)
        parse_sensors(ctl, params)
        parse_ancc_params(ctl, params)

    elif filetype == 'wave-pick':
        pickparams = parse_one(ctl, "pickparams")
        for node in pickparams.childNodes:
            if node.nodeType != node.ELEMENT_NODE:
                continue

            params[node.tagName] = parse_attrib(node, "value", cast=float)

    elif filetype == 'event-location-nlloc':
        parse_grid(ctl, params)
        parse_sensors(ctl, params)
        parse_calc_params(ctl, params)

    elif filetype == 'velocity-modeling':
        parse_grid(ctl, params)
        parse_sensors(ctl, params)
        parse_inversion(ctl, params)

    return params


def buildJob(keys, params):

    # find all the possible combination of input parameters
    args = [params[k] for k in keys]
    return itertools.product(*args)


def getCurrentJobParams(params, keys, job):
    for k, j in zip(keys, job):
        params[k] = j
    return params


def buildMapRedKey(params, keys, job, string=True):
    jobdict = {}
    for i, (k, jb) in enumerate(zip(keys, job)):
        jobdict[k] = jb.index

    if string:
        return str(jobdict)
    else:
        return jobdict


def NLLSuffixFromKey(key):
    """
    Takes a key string representing a dictionary and create the NLL suffix
    """
    jobdict = key

    try:
        suffix = '%d_%d' % (jobdict['velgrids'], jobdict['sensors'])

    except:
        logger.warning('Key not properly constructed suffix will be 0_0')
        suffix = '0_0'

    return suffix
