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
from netCDF4 import Dataset, num2date

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.interpolate import UnivariateSpline


def get_l2p_file_list(l2p_repo):
    """ Convience function to extract l2p files from subdirectory structure """
    l2p_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(l2p_repo)
                 for name in files
                 if name.endswith((".nc"))]
    return sorted(l2p_files)


class MapBaseClass(object):

    default_basemap_dict = dict(projection="stere", lon_0=-45, lat_0=90, 
                                width=6e6, height=6e6, resolution="i")

    def __init__(self, basemap_dict=None):

        # self._set_mpl_defaults()
        if basemap_dict is not None:
            self.basemap_dict = basemap_dict
        else:
            self.basemap_dict = self.default_basemap_dict

    def _init_map(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = plt.gca()
        self.ax.set_position([0.05, 0.05, 0.9, 0.9])
        self.m = Basemap(ax=self.ax, **self.basemap_dict)
        self.m.drawmapboundary(linewidth=0.1, zorder=200)
        self.m.drawcoastlines(linewidth=0.25, color="0.0")
        self.m.fillcontinents(color="#bcbdbf", lake_color="#bcbdbf", zorder=100)

    def set_label(self, label):
        self.ax.set_title(label)

    def set_settings_label(self, label):
        plt.annotate(label, (0.05, 0.045), xycoords="figure fraction", va="top", ha="left")

    def show(self, *args, **kwargs):
        plt.show(*args, **kwargs)

    def savefig(self, *args, **kwargs):
        plt.savefig(*args, **kwargs)


class ThickestIceMap(MapBaseClass):

    def __init__(self, l2p_collect, detection_settings={}, **kwargs):
        super(ThickestIceMap, self).__init__(**kwargs)

        # Initialize the default map
        self._init_map()

        # Plot the thickness ice location
        thickest_lon, thickest_lats = l2p_collect.get_thickest_ice_locations(**detection_settings)
        self.m.scatter(thickest_lon, thickest_lats, c="red", marker="x", s=5, alpha=0.75, latlon=True)

        # Add labels
        self.set_label(l2p_collect.tc_label)
        self.set_settings_label(l2p_collect.settings_label)


class L2pSITMap(MapBaseClass):

    def __init__(self, l2p_collect, **kwargs):
        super(L2pSITMap, self).__init__(**kwargs)

        # Initialize the default map
        self._init_map()

        # Plot the thickness ice location
        for l2p in l2p_collect.profiles: 
            self.m.scatter(l2p.lon, l2p.lat, c=l2p.sit, marker=".", s=1, edgecolors="none",  
                           cmap=plt.get_cmap("plasma"), vmin=0, vmax=5, latlon=True)

        # Add labels
        self.set_label(l2p_collect.tc_label)

class L2PSITCollection(object):

    def __init__(self, l2p_file_list):
        self.l2p_file_list = l2p_file_list
        self.profiles = []

        # Default values for thickest ice detection
        self.sit_threshold = 5.0
        self.smoothed_profile_scaling = 0.8

        self._parse_data()

    def get_thickest_ice_locations(self, sit_threshold=None, smoothed_profile_scaling=None):
        
        # Override default settings with potential keyword arguments
        # NOTE: This is to make sure that the properties in this class are those that
        #       are used for detecting the thickest ice
        thickice_settings = dict(sit_threshold=self.sit_threshold, 
                                 smoothed_profile_scaling=self.smoothed_profile_scaling)
        if sit_threshold is not None:
            self.sit_threshold = sit_threshold
            thickice_settings["sit_threshold"] = sit_threshold
        
        if smoothed_profile_scaling is not None:
            self.smoothed_profile_scaling = smoothed_profile_scaling
            thickice_settings["smoothed_profile_scaling"] = smoothed_profile_scaling

        lons, lats = [], []
        for l2p in self.profiles:
            indices = l2p.get_thickest_ice_indices(**thickice_settings)
            lons.extend(l2p.lon[indices])
            lats.extend(l2p.lat[indices])

        return np.array(lons), np.array(lats)
        
    def _parse_data(self):

        for l2p_file in self.l2p_file_list:
            # Get data from file
            nc = Dataset(l2p_file)
            lon = nc.variables["longitude"][:]
            lat = nc.variables["latitude"][:]
            time = nc.variables["time"]
            times = time[:]
            dt = num2date(times, time.units)
            sit = nc.variables["sea_ice_thickness"][:]
            nc.close()
            self.profiles.append(L2PContainer(times, dt, lon, lat, sit))

    @property
    def tc_label(self):
        """ returns the time coverage label """
        dts_min = np.amin([np.amin(l2p.dt) for l2p in self.profiles])
        dts_max = np.amax([np.amin(l2p.dt) for l2p in self.profiles])
        return "l2p collection: %s - %s" % (dts_min.strftime("%Y-%m-%d"), dts_max.strftime("%Y-%m-%d"))

    @property
    def settings_label(self):
        """ returns the label of the detection settings """
        label = "settings: sit_threshold=%.1fm, smoothed_profile_scaling=%g"
        label = label % (self.sit_threshold, self.smoothed_profile_scaling)
        return label

class L2PContainer(object):

    def __init__(self, time, dt, lon, lat, sit):
        self.time = time
        self.dt = dt
        self.lon = lon
        self.lat = lat
        self.sit = sit

    def get_thickest_ice_indices(self, sit_threshold=5.0, smoothed_profile_scaling=0.8):
        """ Returns the a list of indices that fulfill the thickest sea ice criterium:
        Both mean (smoothing spline) as well as value need to be above the
        a thickness threshold (sit_threshold). The threshold for the smoothed data 
        can be chosen relatively to the absolute thickness threshold
        """

        # 1. Split in orbits
        time_diff = np.ediff1d(self.time)
        new_orbit_indices = np.where(time_diff > 1000.)[0]  # units in seconds

        shape = (len(new_orbit_indices)+1)
        orbit_start_index = np.full(shape, 0)
        orbit_start_index[1:] = new_orbit_indices+1
        
        orbit_end_index = np.full(shape, len(self.time)-1)
        orbit_end_index[:-1] = new_orbit_indices

        # 2. Loop over orbits
        thickest_ice_indices = []
        for i0, i1 in zip(orbit_start_index, orbit_end_index):

            # Compute a univariate spline
            # NOTE: This has been chosen to provide a smooth representation
            #       of the irregular and gappy data coverage
            x, y = self.time[i0:i1+1], self.sit[i0:i1+1]
            valid = np.where(np.isfinite(y))[0]
            spl = UnivariateSpline(x[valid], y[valid])

            # Get the indices where both profile value as well as smoothed thickness
            # profile exceeds a thickness threshold
            y_fit = spl(x[valid])
            local_indices = np.where(
                np.logical_and(y_fit > sit_threshold*smoothed_profile_scaling, 
                               y[valid] > sit_threshold))[0]
            absolute_indices = valid[local_indices] + i0

            # Store the indices
            thickest_ice_indices.extend(absolute_indices)

        return thickest_ice_indices
