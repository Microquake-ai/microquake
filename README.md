# README #

Microquake is an open source package to be licensed under the [GNU general Public License, version 3 (GPLv3)](http://www.gnu.org/licenses/gpl-3.0.html). Microquake is an extension of Obspy for the processing of microseismic data

### Development

```
pip install poetry
poetry install
```

Running tests

```
poetry run pytest
```

### How to release a new version

```
poetry version
git add pyproject.toml
gc -m "bump version"
git tag newversion
git push --tags
```

### Automatic tagging and releasing

By adding the following command to your git config you can bump and release a new version with one command

```
git config --global alias.bump "\!version=\$(poetry version | awk '{print \$NF}' ) && git add pyproject.toml && git commit -m \"Bumping version to \$version\" && git tag \$version && git push --tags"
```

After running the above command you may release a new version with:

```
git bump
```

### Package structure ###

The package structure is not final and will settle over time

#### Content ####

* core: core function similar to Obspy
* db: the database related functions and objects
* doc: documentation
* examples: examples on how to use the library as ipython notebook
* focmec: proto interface to the Focmec software (not working yet)
* imaging: Imaging functions (see Obspy imaging)
* mag: Magnitude calculations
* nlloc: Interface to NonLinLoc (Anthony Lomax)
* plugin: Plugins for reading various formats
* realtime: Real time data processing (see Obspy realtime)
* signal: (see obspy signal)
* simul: Location error and system sensitivity simulation

### What do I need to get microquake running? ###

* Python 3.6+
* Numpy 1.9+
* Scipy 0.14+
* Matplotlib
* [ObsPy 0.9](http://docs.obspy.org/index.html), (version 1+ not supported yet)
* SimpleJSON
* [NonLinLoc 6.0+](http://alomax.free.fr/nlloc/)
* python-dateutil

* pyevtk
* pyvtk
* pyspark
* py4j (for spark)
* SQLAlchemy 0.9+ (only for experimental dB interaction)
* pymongo (only for experimental dB interaction)
* scikit-fmm'  # for the eikonal solver
* mplstereonet' # to plot the station (experimental)

### Contribution guidelines ###

* Text editor - 
* Please use spaces instead of tabs for Python indentation to avoid conflicts
* TODO.org files are in orgmode format. For instance, you can use the **orgmode** package in Sublime Text Package Control for dynamic checkboxes and more!
* Code format adheres as close as possible to [PEP8](https://www.python.org/dev/peps/pep-0008/) style convention. Please use **Flake8Lint** package if using Sublime Text Package Control for automatic highlight of style errors. A custom style template can be found in Flake8Lint.sublime-settings
* Function documentation in code should be in [reST standard](http://stackoverflow.com/questions/5334531/python-documentation-standard-for-docstring)
