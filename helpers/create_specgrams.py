#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create small spectrogram file for each event
:copyright:
    Simon Stähler (mail@simonstaehler.com), 2022
:license:
    None
"""

import obspy
from obspy.signal.tf_misfit import cwt
from helpers import constants as c
import glob
from os.path import join as pjoin
from os.path import split as psplit
import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    event_list = glob.glob('../event_data/*')

    for event_dir in event_list:
        event_path = pjoin(event_dir, 'waveforms', 'waveforms_VBB.mseed')
        event_name = psplit(event_dir)[-1]
        st = obspy.read(event_path)
        st.differentiate()
        st.differentiate()
        st.filter('highpass', freq=1. / 15)
        st.trim(starttime=c.origin_time[event_name],
                endtime=c.origin_time[event_name] + 1800
                )
        for tr in st:
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
            plt.close(fig)
        st.filter('lowpass', freq=2.)
        st.write(f'{event_name}.mseed', format='MSEED')
