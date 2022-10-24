#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Define a few constants that are used throughout
:copyright:
    Simon Stähler (mail@simonstaehler.com), 2022
:license:
    None
"""
from obspy import UTCDateTime as utct

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

event_list = ['S0235b', 'S0173a', 'S0183a', 'S0185a', 'S0809a', 'S1048d', 'S1133c', 'S1094b']

origin_time = dict(
    S0173a=utct('2019-05-23T02:19:09'),
    S0183a=utct('2019-06-03T02:22:01'),
    S0185a=utct('2019-06-05T02:06:37'),
    S0235b=utct('2019-07-26T12:15:38'),
    S0809a=utct('2021-03-07T11:09:26'),
    S1000a=utct('2021-09-18T17:46:20'),
    S1048d=utct('2021-11-07T22:00:15'),
    S1094b=utct('2021-12-24T22:38:02'),
    S1133c=utct('2022-02-03T08:04:38'),
)
