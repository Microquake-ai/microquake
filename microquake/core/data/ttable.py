
import numpy as np
from glob import glob
import os
import h5py


class H5TTable(object):
    """docstring for H5TTable"""

    def __init__(self, path, dset_key=None):
        self.path = path
        self.hf = h5py.File(path, 'r')
        self.keys = list(self.hf.keys())

        self.dset = None

        if dset_key is not None:
            self.set_dataset(dset_key)

        self.stations = self.hf['stations'][:].astype('U4')
        self._stadict = dict(zip(self.stations, np.arange(len(self.stations))))

        self.locations = self.hf['locations'][:]
        self.coords = self.hf['grid_locs'][:]

    def set_dataset(self, key):
        if key in self.keys:
            self.dset = self.hf[key]
        else:
            raise KeyError('dataset %s does not exist' % key)
        
    @property
    def shape(self):
        return self.hf.attrs['shape']

    @property
    def origin(self):
        return self.hf.attrs['origin']

    @property
    def spacing(self):
        return self.hf.attrs['spacing']

    def index_sta(self, stations):
        if isinstance(stations, (list, np.ndarray)):
            return np.array([self._stadict[sta] for sta in stations])
        else:
            return self._stadict[stations]

    def icol_to_xyz(self, index):
        nx, ny, nz = self.shape
        iz = index % nz
        iy = ((index - iz) // nz) % ny
        ix = index // (nz * ny)
        loc = np.array([ix, iy, iz], dtype=float) * self.spacing + self.origin
        return loc

    def xyz_to_icol(self, loc):
        x, y, z = loc
        ix, iy, iz = ((loc - self.origin) / self.spacing).astype(int)
        nx, ny, nz = self.shape
        # return (iz * nx * ny) + (iy * nx) + ix;
        return int((ix * ny * nz) + (iy * nz) + iz)

    def close(self):
        self.hf.close()
      

def gdef_to_points(shape, origin, spacing):
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
    origin = np.array(dat[3:6], dtype=np.float32) * 1000.
    spacing = (np.array(dat[6:9], dtype=np.float32) * 1000.)[0]
    sloc = np.array(dat[12:15], dtype=np.float32) * 1000.

    return sloc, shape, origin, spacing



# f_tt = os.path.join(common_dir, nll_dir, 'time', 'OT.%s.%s.time.buf'
#                             % (phase.upper(), sta_code))

def array_from_nll_grids(path, phase, prefix='OT'):

    phase = phase.upper()
    bufs = os.path.join(path, '%s.%s.*.time.buf' % (prefix, phase))
    headers = os.path.join(path, '%s.%s.*.time.hdr' % (prefix, phase))

    fles = np.sort(glob(bufs))
    hfles = np.sort(glob(headers))
    assert(len(fles) == len(hfles))
    # stations = np.array([f.split('.')[-3].zfill(3) for f in fles], dtype='U4')
    stations = np.array([f.split('.')[-3] for f in fles])
    # stations = np.array([f.split('.')[-3] for f in fles], dtype='S4')
    isort = np.argsort(stations)
    fles = fles[isort]
    hfles = hfles[isort]
    stations = stations[isort]

    vals = [read_nll_header(fle) for fle in hfles]
    sloc, shape, origin, spacing = vals[0]
    slocs = np.array([v[0] for v in vals], dtype=np.float32)
    ngrid = np.product(shape)

    nsta = len(fles)
    tts = np.zeros((nsta, ngrid), dtype=np.float32)

    for i in range(nsta):
        tts[i] = np.fromfile(fles[i], dtype=np.float32)

    # gdef = np.concatenate((shape, origin, [spacing])).astype(np.int32)
    # names = np.array([name.decode('utf-8') for name in names])

    # ndict = {}
    # for i, sk in enumerate(names):
    #     ndict[sk.decode('utf-8')] = i

    # meta = {'spacing': spacing, 'origin': origin, 'shape': shape, 'names': names}
    # data = dict(tts=tts, locations=slocs, origin=origin, spacing=spacing, namedict=ndict)
    data = dict(ttable=tts, locations=slocs, shape=shape, origin=origin, spacing=spacing, stations=stations, phase=phase)

    return data
    # return tts, slocs, ndict, gdef


def write_h5(fname, tdict, tdict2=None):

    # names = np.array(list(ndict.keys()), dtype='S4')
    shape = tdict['shape']
    spacing = tdict['spacing']
    origin = tdict['origin'].astype(np.float32)
    locations = tdict['locations']
    stations = tdict['stations']
    gridlocs = gdef_to_points(shape, origin, spacing)

    hf = h5py.File(fname, 'w')
    hf.attrs['shape'] = shape
    hf.attrs['origin'] = origin
    hf.attrs['spacing'] = spacing
    hf.create_dataset('locations', data=locations.astype(np.float32))
    hf.create_dataset('grid_locs', data=gridlocs.astype(np.float32))

    hf.create_dataset('tt%s' % tdict['phase'].lower(), data=tdict['ttable'])

    if tdict2 is not None:
        hf.create_dataset('tt%s' % tdict2['phase'].lower(), data=tdict2['ttable'])
    
    gdef = np.concatenate((shape, origin, [spacing])).astype(np.int32)
    hf.create_dataset('grid_def', data=gdef)
    hf.create_dataset('stations', data=stations.astype('S4'))
    hf.close()
