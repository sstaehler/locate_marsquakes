#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create small spectrogram file for each event
:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""

color_phases = dict(P='C0',
                    S='C1',
                    PP='C2',
                    SS='C3',
                    ScS='C4')

color_seis = 'darkgrey'
lw_seis = 0.6
color_baz = 'darkblue'
lw_baz = 2.

lon_insight = 135.623447
lat_insight = 4.502384

radius_mars = 3389.5
