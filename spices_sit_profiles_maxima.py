from __future__ import print_function

"""
This script can be used fo the detection of most thickest ice from RA thickness profiles. 
It has been developed as part of the Horizon 2020 project: 

    SPICES - Space-borne observations for detecting and forecasting sea ice cover extremes

The script identifies regions of thickest sea ice using a thickness threshold based on 
daily trajectory (Level-2 preprocessed:l2p) sea ice thickness observations based on the
ESA CryoSat-2 radar altimeter platform. The l2p data can be obtained here: 

    ftp://altim:altim@data.meereisportal.de/altim/sea_ice/product/north/cryosat2/cs2awi-v2.0/l2p_daily/

Usage: 

The script requires the path to a local copy of the l2p data as argument: 

    $ python spices_sit_profiles_maxima.py /path_to_sit_data

The script will ingest all netCDF files within the given path. The user can control time range by
passing selected subfolders, e.g.:

    $ python spices_sit_profiles_maxima.py /path_to_sit_data/l2p_daily/2018/03

"""

import os
import re
import sys
from netCDF4 import Dataset

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.interpolate import UnivariateSpline


def main(l2p_repo):
    """ Add doc here """
    
    # Get a list of thickness profiles
    l2p_file_list = get_l2p_file_list(l2p_repo)
    print("%g l2p files found" % len(l2p_file_list))

    # Create a collection of l2p sea ice thickness data
    l2p_collect = L2PSITCollection(l2p_file_list)

    l2p = l2p_collect.profiles[0]
    l2p.get_maxima()

    # Plot the collection
    create_nh_sit_map(l2p_collect)

def get_l2p_file_list(l2p_repo):
    l2p_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(l2p_repo)
                 for name in files
                 if name.endswith((".nc"))]
    return sorted(l2p_files)

def create_nh_sit_map(l2p_collect):

    plt.figure(dpi=150)
    ax = plt.gca()
    basemap_args = dict(
        projection="stere", lon_0=-45, lat_0=90, 
        width=6e6, height=6e6, resolution="i")
    m = Basemap(ax=ax, **basemap_args)
    m.drawmapboundary(linewidth=0.1, zorder=200)
    m.drawcoastlines(linewidth=0.25, color="0.0")
    m.fillcontinents(color="#bcbdbf", lake_color="#bcbdbf", zorder=100)
    l2p = l2p_collect.profiles[0]
    m.scatter(l2p.lon, l2p.lat, c=l2p.sit, vmin=0, vmax=5, s=1, edgecolors="none",
              cmap=plt.get_cmap("plasma"), latlon=True)
    plt.show()

class L2PSITCollection(object):

    def __init__(self, l2p_file_list):
        self.l2p_file_list = l2p_file_list
        self.profiles = []
        self._parse_data()

    def _parse_data(self):

        for l2p_file in self.l2p_file_list:
            # Get data from file
            nc = Dataset(l2p_file)
            lon = nc.variables["longitude"][:]
            lat = nc.variables["latitude"][:]
            time = nc.variables["time"][:]
            # units = datenum.units
            # time = num2date(datenum[:], units=units)
            sit = nc.variables["sea_ice_thickness"][:]
            nc.close()
            self.profiles.append(L2PContainer(time, lon, lat, sit))


class L2PContainer(object):

    def __init__(self, time, lon, lat, sit):
        self.time = time
        self.lon = lon
        self.lat = lat
        self.sit = sit

    def get_maxima(self, sit_threshold=5.0):

        # 1. Split in orbits
        time_diff = np.ediff1d(self.time)
        new_orbit_indices = np.where(time_diff > 1000.)[0]  # units in seconds

        shape = (len(new_orbit_indices)+1)
        orbit_start_index = np.full(shape, 0)
        orbit_start_index[1:] = new_orbit_indices+1
        
        orbit_end_index = np.full(shape, len(self.time)-1)
        orbit_end_index[:-1] = new_orbit_indices

        plt.figure(dpi=150)
        for i0, i1 in zip(orbit_start_index, orbit_end_index):
            x, y = self.time[i0:i1+1], self.sit[i0:i1+1]
            valid = np.where(np.isfinite(y))[0]
            spl = UnivariateSpline(x[valid], y[valid])
            spl.set_smoothing_factor(0.5)
            plt.scatter(x, y)
            plt.plot(x, spl(x), color='black', lw=3, zorder=100)

        plt.show()
        stop

if __name__ == '__main__':

    # Get the path to the sea ice thickness data repository
    # with some sanity checks
    try: 
        l2p_repo = sys.argv[1]
    except IndexError:
        sys.exit("Error: path to local l2p repository required")
    if not os.path.isdir(l2p_repo):
        raise ValueError("%s is not a valid directory" % str(l2p_repo))

    # Execute the main program 
    main(l2p_repo)