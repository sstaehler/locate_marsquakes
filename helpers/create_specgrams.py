#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create small spectrogram file for each event
:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""

import obspy
from obspy.signal.tf_misfit import cwt
from obspy import UTCDateTime as utct
import glob
from os.path import join as pjoin
from os.path import split as psplit
import numpy as np
import matplotlib.pyplot as plt

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

if __name__=='__main__':
    event_list = glob.glob('../event_data/*')

    for event_dir in event_list:
        event_path = pjoin(event_dir, 'waveforms', 'waveforms_VBB.mseed')
        event_name = psplit(event_dir)[-1]
        st = obspy.read(event_path)
        st.differentiate()
        st.differentiate()
        st.filter('highpass', freq=1. / 15)
        st.trim(starttime=origin_time[event_name],
                endtime=origin_time[event_name] + 1800
                )
        for tr in st:
            # tr.trim(starttime=tr.stats.starttime, endtime=tr.stats.endtime-800.
            #         )
            npts = tr.stats.npts
            dt = tr.stats.delta
            t = np.linspace(0, dt * npts, npts)
            f_min = 0.05
            f_max = 10
            scalogram = cwt(tr.data, dt, 12, f_min, f_max)
            f = np.geomspace(f_min, f_max, scalogram.shape[0])
            dt = 5
            df = 2
            np.savez(f'{event_name}_spec_{tr.id}.npz', f=f[::df], t=t[::dt],
                     spec=np.abs(scalogram[::df, ::dt]))
            fig, ax = plt.subplots(1, 1)
            ax.pcolormesh(t[::dt], f[::df], np.log10(np.abs(scalogram[::df, ::dt])))
            ax.set_yscale('log')
            plt.savefig(f'{event_name}_spec_{tr.id}.png')
        st.filter('lowpass', freq=2.)
        st.write(f'{event_name}.mseed', format='MSEED')
