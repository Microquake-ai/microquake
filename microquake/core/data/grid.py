from scipy.interpolate import griddata
from scipy.ndimage.interpolation import map_coordinates
from microquake.core import logger
from pkg_resources import load_entry_point

from microquake.core.util import ENTRY_POINTS

import numpy as np

# from pyevtk.hl import imageToVTK


def GenEventsOnGrid(Grid, ev_spacing):
    import numpy as np
    cmin = np.array(Grid.origin)

    cmax = cmin + np.array(Grid.shape) * Grid.spacing

    nele = np.floor((cmax - cmin) / ev_spacing)

    x = np.linspace(cmin[0], cmax[0], nele[0])
    y = np.linspace(cmin[1], cmax[1], nele[1])
    z = np.linspace(cmin[2], cmax[2], nele[2])

    Y, X, Z = np.meshgrid(y, x, z)
    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()

    Y, X, Z = np.meshgrid(y, x, z)

    coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    return coords


def create(method, **kwargs):
    if method == 'ODS':
        return _createODS(**kwargs) 
    elif method == 'OCS':
        return _createOCS(**kwargs)


def _createODS(origin=None, dimensions=None, spacing=None, val=0, **kwargs):
    """
    create a grid from origin, dimensions and spacing
    :param origin: grid origin
    :type origin: tuple
    :param dimensions: grid dimension
    :type dimensions: tuple
    :param spacing: spacing between the grid nodes
    :type spacing: float
    :param val: constant value with which to fill the grid
    :rtype: ~microquake.core.data.grid.GridData
    """
    import numpy as np
    data = np.ones(tuple(dimensions)) * val
    grid = GridData(data, spacing=spacing, origin=origin)
    return grid


def _createOCS(origin=None, corner=None, spacing=None, val=0, buf=0, **kwargs):
    """
    create a grid from origin, dimensions and spacing
    :param origin: grid origin (e.g., lower left corner for 2D grid)
    :type origin: tuple
    :param corner: grid upper (e.g., upper right corner for 2D grid)
    :type corner: tuple
    :param spacing: spacing between the grid nodes
    :type spacing: float
    :param val: constant value with which to fill the grid
    :param buf: buffer around the grid in fraction of grid size
    :rtype: ~microquake.core.data.grid.GridData
    """
    import numpy as np
    tmp = buf * (corner - origin)
    origin2 = origin - tmp
    corner2 = corner + tmp

    gshape = tuple([int(np.ceil((c - o) / spacing)) for o, c in zip(origin2, corner2)])
    data = np.ones(gshape) * val
    grid = GridData(data, spacing=spacing, origin=origin)
    grid.fill_homogeneous(val)

    return grid


def ones(shape, origin=(0, 0, 0), spacing=1, **kwargs):
    """
    Return a new GridData object of given shape, filled with ones. For extra
    parameter see numpy.ones
    :param shape: Shape of the new array, e.g., ``(2, 3)`` or ``2``
    :type shape: int or a sequence of int
    :param origin: grid origin
    :type: int or a sequence of int
    :param spacing: node spacing (the node spacing is the same in every
    dimension)
    :type spacing: int
    :rtype: ~microquake.core.data.grid.GridData
    """

    import numpy as np

    data = np.ones(shape)
    return GridData(data, origin=origin, spacing=spacing)
    
# def load(fle, format="NLL", **kwargs):
#     if format == 'NLL':
#         return _loadNLL(fle, **kwargs)


def read_grid(filename, format='PICKLE', **kwargs):
    """
    read a grid
    :param filename: the name of the file
    :param format: format of the file
    :return: GridData object
    """
    return readGrid(filename, format=format, **kwargs)


def readGrid(filename, format='PICKLE', **kwargs):
    format = format.upper()
    if format not in ENTRY_POINTS['grid'].keys():
        logger.error('Grid format %s is not currently supported' % format)
        return

    format_ep = ENTRY_POINTS['grid'][format]
    read_format = load_entry_point(format_ep.dist.key,
            'microquake.plugin.grid.%s' % format_ep.name, 'readFormat')

    return read_format(filename, **kwargs)


def readBufferOffsets(self, point):
    import numpy as np

    ev_offset_grid = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                      [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    ev_offset_grid = np.array(ev_offset_grid)

    # array of 8 XYZ points defining the corners of a cube
    points = point + ev_offset_grid * self.spacing

    mod_index = self.transform_to(point)
    interp_needed = not np.all(mod_index == mod_index.astype(int))
    mod_index = mod_index.astype(int)

    # determine offsets in the grid file to read the 8 points (if necessary)
    offsets = []
    if interp_needed:

        # index is outside of the grid
        if np.any(mod_index < 0) or np.any((self.shape - mod_index) < 1):
            return

        # index is on the upper edge of the grid
        if np.any((self.shape - mod_index) == 1):
            edge_index = np.nonzero((self.shape - mod_index) == 1)[0]
            for e in edge_index:
                mod_index[e] -= 1

        for o in ev_offset_grid:
            byte_offset = np.ravel_multi_index(mod_index + o, dims=self.shape, order='C') * 4
            offsets.append(byte_offset)
    else:
        byte_offset = np.ravel_multi_index(mod_index, dims=self.shape, order='C') * 4
        offsets.append(byte_offset)

    return points, offsets, interp_needed


def readBuffer(self, xi, points, offsets, interp_needed, grid_file):
    import numpy as np

    values = []  # array of the values of the grid defined at the 8 points
    for o in offsets:
        # logger.debug(o)
        fpo = np.memmap(grid_file, dtype='f4', mode='r', offset=o, shape=(1,))
        values.append(fpo[0])

    if interp_needed:
        data = griddata(points, values, xi, method='linear')
    else:
        data = values[0]

    return data


def homogenous_like(grid_data, value=1):
    """
    create an homogeneous grid from a GridData object (grid_data) with the same
    shape, spacing and origin
    :param grid_data: an existing GridData object
    :type grid_data: ~microquake.core.data.grid.GridData
    :param value: the value with which to fill the grid object (default 1)
    :rtype: ~microquake.core.data.grid.GridData
    """
    out_grid = grid_data.copy()
    out_grid.data.fill(value)
    return out_grid


def read_grid_collection(file_path, format="HDF5"):
    """
    read GridCollection from file
    :param filename: file path
    :param format: format of the file (default "HDF5")
    :return: GridCollection
    """
    pass


class GridData(object):
    """
    object containing a grid define at regularly spaced node. Note that the
    spacing is the same for every dimension

    :param shape_or_data: a numpy array or the shape of the underlying data
    :param spacing: The spacing of the grid
    :type spacing: float
    :param origin: A Tuple representing the position of the lower left corner of
    the grid
    :type origin: tuple
    :param seed_label: seed identifyer can be a station code
    :type station_code: str
    :param seed: A Tuple representing the seed location
    :type seed: tuple
    """

    def __init__(self, data, spacing=1, origin=None,
                 seed_label=None, seed=None, grid_type='VELOCITY'):
        import numpy as np
        self.data = []
        data, np.ndarray
        origin = np.array(origin)
        self.data = data
        if origin is None:
            self.origin = np.zeros(len(data.shape))
        else:
            if origin.shape[0] == len(data.shape):
                self.origin = origin
            else:
                self.origin = np.zeros(len(data.shape))
                # print "origin shape does not match data shape\n origin set to 0"
        self.spacing = spacing
        self.seed_label = seed_label
        self.seed = seed
        self.type = grid_type.upper()

    def __setattr__(self, attr, value):
        from microquake.nlloc import valid_nlloc_grid_type
        if attr == "type":
            self.__dict__[attr] = str(value).upper()
            if value not in valid_nlloc_grid_type:
                logger.warning('grid type provided is not a valid nlloc grid'
                'type')
        self.__dict__[attr] = value

    def __repr__(self):
        repr_str =  """
        spacing: %s
        origin : %s
        shape  : %s
        seed   : %s
        type   : %s
        """ % (self.spacing, self.origin, self.shape, self.seed, self.type)
        return repr_str

    def __mul__(self, other):
        if isinstance(other, GridData):
            if self.check_compatibility(other):
                return self.data * other.data
        else:
            try:
                return self.data * other
            except:
                pass

    def transform_to(self, values):
        """
        transform model space coordinates into grid space coordinates
        :param values: tuple of model space coordinates
        :type values: tuple
        :rtype: tuple
        """
        from numpy import array
        coords = (values - self.origin)/self.spacing

        return coords

    def transform_from(self, values):
        """
        transform grid space coordinates into model space coordinates
        :param values: tuple of grid space coordinates
        :type values: tuple
        :rtype: tuple
        """
        return values * self.spacing + self.origin

    def check_compatibility(self, other):
        """
        check if two grids are compatible, i.e., have the same shape, spacing
        and origin
        """
        from numpy import all
        return (self.shape == other.shape) and \
               (self.spacing == other.spacing) and \
                all(self.origin == other.origin)

    def __get_shape__(self):
        """
        return the shape of the object
        """
        return self.data.shape

    shape = property(__get_shape__)

    def copy(self):
        """
        copy the object using copy.deepcopy
        """
        import copy
        cp = copy.deepcopy(self)
        return cp

    def in_grid(self, point):
        """
        Check if a point is inside the grid
        :param point: the point to check
        :type point: tuple, list or numpy array
        :returns: True if point is inside the grid
        :rtype: bool
        """
        from numpy import array, all
        corner1 = self.origin
        corner2 = self.origin + self.spacing * array(self.shape)

        return all((point >= corner1) & (point <= corner2))

    def fill_homogeneous(self, value):
        """
        fill the data with a constant value
        :param value: the value with which to fill the array
        """
        self.data.fill(value)

    def generate_points(self, pt_spacing=None):
        """
        """
        import numpy as np
        # if pt_spacing is None:
        ev_spacing = self.spacing

        dimensions = np.array(self.shape) * self.spacing / ev_spacing

        xe = np.arange(0, dimensions[0]) * ev_spacing + self.origin[0]
        ye = np.arange(0, dimensions[1]) * ev_spacing + self.origin[1]
        ze = np.arange(0, dimensions[2]) * ev_spacing + self.origin[2]

        Xe, Ye, Ze = np.meshgrid(xe, ye, ze)

        Xe = Xe.reshape(np.prod(Xe.shape))
        Ye = Ye.reshape(np.prod(Ye.shape))
        Ze = Ze.reshape(np.prod(Ze.shape))
        return Xe, Ye, Ze

    def write(self, filename, format='PICKLE', **kwargs):
        """
        write the grid to disk
        :param filename: full path to the file to be written
        :type filename: str
        :param format: output file format
        :type format: str
        """
        format = format.upper()
        if format not in ENTRY_POINTS['grid'].keys():
            logger.error('format %s is not currently supported for Grid '
                         'objects' % format)
            return

        format_ep = ENTRY_POINTS['grid'][format]
        write_format = load_entry_point(format_ep.dist.key,
                'microquake.plugin.grid.%s' % format_ep.name, 'writeFormat')

        write_format(self, filename, **kwargs)

    def interpolate(self, coord, grid_coordinate=True, mode='nearest', *args,
                    **kwargs):
        """
        This function interpolate the values at a given point expressed either in grid or absolute coordinates
        :param coord: Coordinate of the point(s) at which to interpolate either in grid or absolute coordinates
        :type coord: list, tuple, numpy.array
        :param grid_coordinate: whether the coordinates are provided in grid coordinates or not
        :type grid_coordinate: bool
        :rtype: numpy.array
        """
        import numpy as np

        coord = np.array(coord)

        if not grid_coordinate:
            coord = self.transform_to(coord)

        if len(coord.shape) < 2:
            coord = coord[:,np.newaxis]

        try:
            return map_coordinates(self.data, coord, mode=mode, *args, **kwargs)
        except:
            return map_coordinates(self.data, coord.T, mode=mode, *args,
                                   **kwargs)

