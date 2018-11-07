#!/usr/bin/env bash

# General development/build
sudo apt-get install build-essential python-dev --assume-yes

# Compilers/code integration
sudo apt-get install gfortran --assume-yes
sudo apt-get install swig --assume-yes

# Numerical/algebra packages
sudo apt-get install libatlas-dev --assume-yes
sudo apt-get install liblapack-dev --assume-yes

# Fonts (for matplotlib)
sudo apt-get install libfreetype6 libfreetype6-dev --assume-yes

# More fonts (for matplotlib on Ubuntu Server 14.04)
sudo apt-get install libxft-dev --assume-yes

# Graphviz for pygraphviz, networkx, etc.
sudo apt-get install graphviz libgraphviz-dev --assume-yes

# Python require pandoc for document conversions, printing, etc.
sudo apt-get install pandoc --assume-yes

# Tinkerer dependencies
sudo apt-get install libxml2-dev libxslt-dev zlib1g-dev --assume-yes

# libpng
sudo apt-get install libpng-dev

sudo apt-get install python-pip --assume-yes

# for pyqt4 and the user interface
sudo apt-get install python-pip python2.7-dev libxext-dev python-qt4 \
qt4-dev-tools build-essential --assume-yes

# installing Focmec