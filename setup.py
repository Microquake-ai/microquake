#!/usr/bin/env python

import os
import sys

from setuptools import setup

version = "0.1.0"

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    os.system('python setup.py bdist_wheel upload')
    sys.exit()

if sys.argv[-1] == 'tag':
    os.system("git tag -a %s -m 'version %s'" % (version, version))
    os.system("git push --tags")
    sys.exit()

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    # 'SQLAlchemy',  # for interaction with SQL database
    'cycler',
    'future',
    'IPython',
    'lxml',
    'numpy',
    'scipy',
    'matplotlib',
    'pyparsing',
    'python-dateutil',
    'pytz',  # time zone support
    'obspy',
    # 'memoize',
    # 'pyevtk',
    # 'pyvtk',  # for reading and writing grids and event location in vtk format
    # 'bson',  # for interaction with mongodb installed with pymongo
    'pymongo',  # for the db module and interaction with mongodb
    # 'py4j',  # for spark 
    'scikit-fmm',  # for the eikonal solver
    # 'mplstereonet', # to plot the station
    # 'pandas'
    #'libcomcat' # for manipulating error ellipsoid
]

# http://blog.prabeeshk.com/blog/2014/10/31/install-apache-spark-on-ubuntu-14-dot-04/

long_description = readme + '\n\n' + history

if sys.argv[-1] == 'readme':
    print(long_description)
    sys.exit()

# from obspy.core.util.base import ENTRY_POINTS

# for console_script in [
#         'MQ-simulation = microquake.core.scripts.simulation:main',
#         'MQ-autoprocess = microquake.core.scripts.autoprocess:main',
#         'MQ-init_project = microquake.core.scripts.init_project:main',
#         'MQ-init_db = microquake.core.scripts.init_db:main',
#         'MQ-import_ESG_SEGY = microquake.core.scripts.import_ESG_SEGY:main',
#         'MQ-wave = microquake.ui.picker.picker:picker'
#         ]:
#     ENTRY_POINTS['console_scripts'].append()
#
# for waveform in ['ESG_SEGY = io.waveform',
#                  'HSF = io.waveform',
#                  'TEXCEL_CSV = io.waveform',
#                  'IMS_CONTINUOUS = io.waveform',
#                  'IMS_ASCII = io.waveform']:
#     ENTRY_POINTS['waveform'].append(waveform)
#
# ENTRY_POINT['event']['Q']


ENTRY_POINTS = {
    'console_scripts': [
        'MQ-simulation = microquake.core.scripts.simulation:main',
        'MQ-autoprocess = microquake.core.scripts.autoprocess:main',
        'MQ-init_project = microquake.core.scripts.init_project:main',
        'MQ-init_db = microquake.core.scripts.init_db:main',
        'MQ-import_ESG_SEGY = microquake.core.scripts.import_ESG_SEGY:main',
        'MQ-wave = microquake.ui.picker.picker:picker'
        ],
    'microquake.io.waveform': [
        'ESG_SEGY = microquake.io.waveform',
        'HSF = micorquake.io.waveform',
        'TEXCEL_CSV = microquake.io.waveform',
        'IMS_CONTINUOUS = microquake.io.waveform',
        'IMS_ASCII = microquake.io.waveform'
        ],

    'microquake.io.event': [
        'QUAKEML = microquake.io.quakeml',
        'NLLOC = microquake.io.nlloc'
        ],

    'microquake.io.waveform.ESG_SEGY': [
        'readFormat = microquake.io.waveform:read_ESG_SEGY'
        ],

    'microquake.io.waveform.HSF': [
        'readFormat = microquake.io.waveform:read_HSF'
        ],

    'microquake.io.waveform.TEXCEL_CSV': [
        'readFormat = microquake.io.waveform:read_TEXCEL_CSV'
        ],

    'microquake.io.waveform.IMS_CONTINUOUS': [
        'readFormat = microquake.io.waveform:read_IMS_CONTINUOUS'
        ],

    'microquake.io.waveform.IMS_ASCII': [
        'readFormat = microquake.io.waveform:read_IMS_ASCII'
    ],

    'microquake.io.nlloc.NLLOC': [
        'NLLOC = microquake.io.nlloc'
    ],

    'microquake.io.nlloc.NLLOC': [
        'readFormat = microquake.plugin.waveform:read_nlloc_hypo'
    ],

    # 'microquake.io.quakeml.QUAKEML': [
    #     'QUAKEML = microquake.io.quakeml'
    # ],
    #
    # 'microquake.io.quakeml.QUAKEML':[
    #     'readFormat = microquake.io.quakeml.core:_read_quakeml'
    # ],

    'microquake.plugin.grid': [
        'NLLOC = microquake.plugin.grid',
        'VTK = microquake.plugin.grid',
        'PICKLE = microquake.plugin.grid'
    ],
     'microquake.plugin.site': [
        'CSV = microquake.plugin.site',
        'VTK = microquake.plugin.site',
        'PICKLE = microquake.plugin.site'
    ],
     'microquake.plugin.grid.NLLOC': [
        'readFormat = microquake.plugin.grid:read_nll',
        'writeFormat = microquake.plugin.grid:write_nll'
    ],
    'microquake.plugin.grid.VTK': [
        'writeFormat = microquake.plugin.grid:write_vtk'
    ],
    'microquake.plugin.grid.PICKLE': [
        'readFormat = microquake.plugin.grid:read_pickle',
        'writeFormat = microquake.plugin.grid:write_pickle'
    ],
    'microquake.plugin.site.CSV': [
        'readFormat = microquake.plugin.site:read_csv',
        'writeFormat = microquake.plugin.site:write_csv'
    ],
    'microquake.plugin.site.PICKLE': [
        'readFormat = microquake.plugin.site:read_pickle',
        'writeFormat = microquake.plugin.site:write_pickle'
    ],
    'microquake.plugin.site.VTK': [
        'writeFormat = microquake.plugin.site:write_vtk'
    ],
    'microquake.io.nlloc': [
        'NLLOC = microquake.io.nlloc'
    ],
    'microquake.plugin.event.NLLOC': [
        'readFormat = microquake.io.nlloc.core:read_nll_event_file'
    ],
}

setup(
    name='microquake',
    version=version,
    description=('Python library that is an extension/expansion/adaptation of'
                'ObsPy to microseismic data'),
    long_description=long_description,
    author='microquake development team',
    author_email='devs@microquake.org',
    url='https://jpmercier@bitbuket.com/microquake',
    packages=[
        'microquake',
    ],
    package_dir={'microquake': 'microquake'},
    entry_points=ENTRY_POINTS,
    include_package_data=True,
    install_requires=requirements,
    license='GNU Lesser General Public License, Version 3 (LGPLv3)',
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering',
    ],
    keywords=(
        'microquake, seismology, mining, microseismic, signal processing, '
        'event location, 3D velocity, automatic, processing, Python, '
        'focal mechanism'
    ),
)
