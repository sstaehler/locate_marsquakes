#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2020
:license:
    None
"""
from taup_distance.taup_distance import get_dist
from obspy.taup import TauPyModel
import numpy as np

model_list = ('iasp91', './taup_tmp/model_example.npz')
for model_file in model_list:
    print('Test with %s' % model_file)
    model = TauPyModel(model_file)
    tSmP = 220.
    dist = get_dist(model=model, tSmP=tSmP, depth=50.)
    arrivals = model.get_travel_times(source_depth_in_km=50., distance_in_degree=dist,
                                      phase_list=('P', 'S'))
    has_P = False
    has_S = False
    for arr in arrivals:
        if arr.name == 'P' and not has_P:
            tP = arr.time
            has_P = True
        if arr.name == 'S' and not has_S:
            tS = arr.time
            has_S = True

    np.testing.assert_almost_equal(actual = tS - tP, desired = tSmP, decimal=3)
    print(dist)

