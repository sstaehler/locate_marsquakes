#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create array with lines of constant baz and distance
:copyright:
    Simon Stähler (mail@simonstaehler.com), 2022
:license:
    None
"""

from locate_2 import shoot
import constants as c
import numpy as np

if __name__ == '__main__':
    ndist = 30
    nazi = 60
    dists = np.linspace(1, 85, ndist)
    azis = np.linspace(0, 360, nazi, endpoint=True)
    lat = np.zeros((ndist, nazi))
    lon = np.zeros((ndist, nazi))
    for idist, dist in enumerate(dists):
        for iazi, azi in enumerate(azis):
            lat[idist, iazi], lon[idist, iazi] = shoot(latitude_1_degree=c.lat_insight,
                                                       longitude_1_degree=c.lon_insight,
                                                       bearing_degree=azi, distance_km=np.deg2rad(dist) * c.radius_mars,
                                                       radius_km=c.radius_mars)

    lon = lon % 360.
    np.savez_compressed('equidistances.npz', dists=dists, azis=azis, lats=lat, lons=lon)
