#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create npz file with travel time curves
:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""
from obspy.taup import TauPyModel
import numpy as np
import json


model = TauPyModel('../model_data/model4taup10.npz')
TT = dict(P=[],
          S=[],
          PP=[],
          SS=[],
          ScS=[])
for dist in np.arange(0, 180, 1):
    for phase, _ in TT.items():
        t = model.get_travel_times(source_depth_in_km=20,
                                   distance_in_degree=dist,
                                   phase_list=[phase]
                                   )
        if len(t) > 0:
            TT[phase].append((dist, t[0].time))
TT_array = dict()
for phase, tt in TT.items():
    TT_array[phase] = np.asarray(tt)

np.savez('traveltimes.npz', P=TT['P'], S=TT['S'], PP=TT['PP'], ScS=TT['ScS'], SS=TT['SS'])