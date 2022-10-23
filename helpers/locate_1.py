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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from ipywidgets import widgets
from taup_distance import taup_distance
from obspy.taup import TauPyModel
from helpers import constants as c
import matplotlib.image as mpimg
from helpers.create_specgrams import origin_time


class Locate_1(widgets.HBox):
    def __init__(self):
        super().__init__()
        plt.close('all')
        output = widgets.Output()
        self.model = TauPyModel('model_data/InSight_KKS21B.npz')
        self.TT = np.load('helpers/traveltimes.npz')

        self.all_st = dict()
        self.all_spec_all = dict()
        self.all_f_spec = dict()
        self.all_t_spec = dict()

        for event in origin_time.keys():
            st = obspy.read(f'helpers/{event}.mseed')
            st.integrate()

            spec_all = dict()
            for comp in ['Z', 'N', 'E']:
                data = np.load(f'helpers/{event}_spec_XB.ELYSE.02.BH{comp}.npz')
                f_spec = data['f']
                t_spec = data['t']
                spec_all[comp] = data['spec']

            self.all_f_spec[event] = f_spec
            self.all_t_spec[event] = t_spec
            self.all_spec_all[event] = spec_all
            self.all_st[event] = st
        initial_event = 'S0235b'
        initial_tP = 150
        initial_tS = 400
        self.tP = initial_tP
        self.tS = initial_tS
        self.plot_spec = False

        with output:
            self.initialize_figure(initial_tP, initial_tS)
            self.h_dotS = None  # ax_dist.plot(dist, t_S_theo, 'o')
            self.h_dotP = None  # ax_dist.plot(dist, t_P_theo, 'o')
            self.h_line = None  # ax_dist.plot([dist, dist], [t_P_theo, t_S_theo], 'k')
            self.h_line_cont = None

            # Plot initial seismogram
            self.l_seis = dict()
            self.h_spec = dict()

            self.set_event(initial_event)

            for tr, ax, comp in zip(self.st, (self.ax_Z, self.ax_N, self.ax_E),
                                    ('Z', 'N', 'E')):
                self.l_seis[comp] = ax.plot(tr.times(), tr.data * 1e9,
                                            lw=c.lw_seis, c=c.color_seis)
                ax.set_xlim(100, 900)

                ax.set_ylabel('vel. / nm/s')
                self.h_spec[comp] = None
            '''        
            ax_time.set_xlim(utct(origin_time[event] + 100).datetime, 
                             utct(origin_time[event] + 900).datetime)
            '''

        # create some control elements
        tP_slider = widgets.IntSlider(value=initial_tP,
                                      min=0, max=1200,
                                      step=1, description='P-arrival')
        tS_slider = widgets.IntSlider(value=initial_tS,
                                      min=0, max=1200,
                                      step=1, description='S-arrival')
        event_combobox = widgets.Dropdown(
            value=initial_event,
            options=['S0235b', 'S0173a', 'S0185a', 'S1094b'],
            description='Event'
        )
        spec_checkbox = widgets.Checkbox(
            description='Plot spectrograms?',
            value=False
        )

        tP_slider.observe(self.update_tP, 'value')
        tS_slider.observe(self.update_tS, 'value')
        event_combobox.observe(self.update_event, 'value')
        spec_checkbox.observe(self.update_spec, 'value')

        controls_1 = widgets.HBox([tP_slider, tS_slider])

        controls_2 = widgets.HBox([event_combobox, spec_checkbox])
        controls = widgets.VBox([controls_1, controls_2])
        # controls.layout = make
        # widgets.HBox([controls, output])
        # add to children
        self.children = [controls, output]

    def set_event(self, initial_event):
        self.event = initial_event
        self.f_spec = self.all_f_spec[self.event]
        self.t_spec = self.all_t_spec[self.event]
        self.spec_all = self.all_spec_all[self.event]
        self.st = self.all_st[self.event]

    def initialize_figure(self, initial_tP, initial_tS):
        self.fig = plt.figure(figsize=(8, 12), constrained_layout=True)
        gs = GridSpec(6, 5, figure=self.fig)
        self.ax_Z = self.fig.add_subplot(gs[0, 0:3])
        self.ax_N = self.fig.add_subplot(gs[1, 0:3], sharex=self.ax_Z)
        self.ax_E = self.fig.add_subplot(gs[2, 0:3], sharex=self.ax_Z)
        self.ax_dist = self.fig.add_subplot(gs[0:2, 3:])
        # Plot for distance
        self.ax_map = self.fig.add_subplot(gs[3:, :])
        # self.ax_map.set_xlim(-90., 90.)
        self.ax_map.set_ylim(-90., 90.)
        self.ax_map.set_ylabel('latitude')
        self.ax_map.set_aspect('equal', 'box')
        self.ax_map.yaxis.tick_right()
        self.ax_map.yaxis.set_label_position('right')
        self.h_circ = None

        img = mpimg.imread('./helpers/MOLA_rolled.png')

        self.ax_map.imshow(img, extent=(-0, 360, -90, 90), cmap='gist_earth')
        self.ax_map.set_xlim(060., 260.)
        self.ax_Z.set_xlim(50, 200)
        self.ax_E.set_xlabel('time / second')
        self.l_P_Z = self.ax_Z.axvline(initial_tP, c=c.color_phases['P'])
        self.l_P_N = self.ax_N.axvline(initial_tP, c=c.color_phases['P'], ls='dashed')
        self.l_P_E = self.ax_E.axvline(initial_tP, c=c.color_phases['P'], ls='dashed')
        self.l_S_Z = self.ax_Z.axvline(initial_tS, c=c.color_phases['S'], ls='dashed')
        self.l_S_N = self.ax_N.axvline(initial_tS, c=c.color_phases['S'])
        self.l_S_E = self.ax_E.axvline(initial_tS, c=c.color_phases['S'])
        for phase in ['P', 'S', 'PP', 'SS', 'ScS']:
            t = np.asarray(self.TT[phase])
            self.ax_dist.plot(t[:, 0], t[:, 1], label=phase, c=c.color_phases[phase])
        self.ax_dist.set_xlim(0, 100)
        self.ax_dist.set_ylim(0, 1000)
        self.ax_dist.legend()
        self.ax_dist.xaxis.tick_top()
        self.ax_dist.xaxis.set_label_position('top')
        self.ax_dist.yaxis.tick_right()
        self.ax_dist.yaxis.set_label_position('right')
        self.ax_dist.set_xlabel('distance')
        self.ax_dist.set_ylabel('t$_S$ - t$_P$')

    # callback functions
    def update_distance(self, tP, tS):  # , h_dotP, h_dotS, h_line):
        # global h_dotP, h_dotS, h_line, h_line_cont, h_circ
        if self.h_dotP is not None:
            [h.remove() for h in self.h_dotP]
        if self.h_dotS is not None:
            [h.remove() for h in self.h_dotS]
        if self.h_line is not None:
            [h.remove() for h in self.h_line]
        if self.h_line_cont is not None:
            [h.remove() for h in self.h_line_cont]
        if self.h_circ is not None:
            self.h_circ.remove()  # [h.remove() for h in h_circ]

        dist = taup_distance.get_dist(model=self.model, tSmP=tS - tP, depth=50)

        if dist is not None:
            t_P_theo = self.model.get_travel_times(distance_in_degree=dist,
                                                   source_depth_in_km=50,
                                                   phase_list=['P'])[0].time
            t_S_theo = self.model.get_travel_times(distance_in_degree=dist,
                                                   source_depth_in_km=50,
                                                   phase_list=['S'])[0].time
            self.h_dotS = self.ax_dist.plot(dist, t_S_theo, 'o', c=c.color_phases['S'])
            self.h_dotP = self.ax_dist.plot(dist, t_P_theo, 'o', c=c.color_phases['P'])
            self.h_line = self.ax_dist.plot([dist, dist], [t_P_theo, t_S_theo], 'k')
            self.h_line_cont = self.ax_dist.plot([dist, dist], [t_P_theo, 1000], 'k--')

            circ = Circle(xy=(135., 3.5), radius=dist, ec='k', fill=False)
            self.h_circ = self.ax_map.add_patch(circ)

            self.ax_map.set_xlim(060., 260.)
            self.ax_map.yaxis.set_label_position('right')


        else:
            self.h_dotS = None  # ax_dist.plot(dist, t_S_theo, 'o')
            self.h_dotP = None  # ax_dist.plot(dist, t_P_theo, 'o')
            self.h_line = None  # ax_dist.plot([dist, dist], [t_P_theo, t_S_theo], 'k')
            self.h_line_cont = None

    def update_tP(self, change):
        # global tP, tS
        """redraw line (update plot)"""
        self.l_P_Z.set_xdata([change.new, change.new])
        self.l_P_N.set_xdata([change.new, change.new])
        self.l_P_E.set_xdata([change.new, change.new])
        self.update_distance(tP=change.new, tS=self.tS)  # ,
        # h_dotP=h_dotP, h_dotS=h_dotS, h_line=h_line)
        self.fig.canvas.draw()
        self.tP = change.new

    def update_tS(self, change):
        # global tP, tS
        """redraw line (update plot)"""
        self.l_S_Z.set_xdata([change.new, change.new])
        self.l_S_E.set_xdata([change.new, change.new])
        self.l_S_N.set_xdata([change.new, change.new])
        self.update_distance(tP=self.tP, tS=change.new)  # ,
        # h_dotP=h_dotP, h_dotS=h_dotS, h_line=h_line)
        self.fig.canvas.draw()
        self.tS = change.new

    def update_event(self, change):
        # global f_spec, t_spec, spec_all, event, plot_spec
        self.set_event(change.new)

        if self.plot_spec:
            for comp, ax in zip(['Z', 'N', 'E'], (self.ax_Z, self.ax_N, self.ax_E)):

                if self.h_spec[comp] is not None:
                    self.h_spec[comp].remove()
                spec = 20 * np.log10(self.spec_all[comp])
                self.h_spec[comp] = ax.pcolormesh(self.t_spec, self.f_spec, spec,
                                                  vmin=-210,
                                                  vmax=np.percentile(spec, q=90))
                ax.set_yscale('log')
                ax.set_ylim(0.05, 10)
                ax.set_ylabel('frequency / Hz')
        else:
            # [h.remove() for h in h_spec[comp]]
            for tr, comp, ax in zip(self.st, ['Z', 'N', 'E'], (self.ax_Z, self.ax_N, self.ax_E)):
                [l.remove() for l in self.l_seis[comp] if l is not None]
                ax.set_yscale('linear')
                self.l_seis[comp] = ax.plot(tr.times(), tr.data * 1e9,
                                            lw=c.lw_seis, c=c.color_seis)
                ylim = np.percentile(abs(tr.data * 1e9), q=98)
                ax.set_ylim([-ylim * 1.5, ylim * 1.5])
                ax.set_ylabel('vel. / nm/s')

    def update_spec(self, change):
        # global f_spec, t_spec, spec_all, plot_spec
        self.plot_spec = change.new
        self.f_spec = self.all_f_spec[self.event]
        self.t_spec = self.all_t_spec[self.event]
        self.spec_all = self.all_spec_all[self.event]
        self.st = self.all_st[self.event]

        if self.plot_spec:
            for comp, ax in zip(['Z', 'N', 'E'], (self.ax_Z, self.ax_N, self.ax_E)):
                [l.remove() for l in self.l_seis[comp] if l is not None]
                spec = 20 * np.log10(self.spec_all[comp])
                self.h_spec[comp] = ax.pcolormesh(self.t_spec, self.f_spec, spec,
                                                  vmin=-210,
                                                  vmax=np.percentile(spec, q=90))
                ax.set_yscale('log')
                ax.set_ylim(0.05, 10)
                ax.set_ylabel('frequency / Hz')
        else:
            # [h.remove() for h in h_spec[comp]]
            for tr, comp, ax in zip(self.st, ['Z', 'N', 'E'], (self.ax_Z, self.ax_N, self.ax_E)):
                if self.h_spec[comp] is not None:
                    self.h_spec[comp].remove()
                ax.set_yscale('linear')
                self.l_seis[comp] = ax.plot(tr.times(), tr.data * 1e9,
                                            lw=c.lw_seis, c=c.color_seis)
                ylim = np.percentile(abs(tr.data * 1e9), q=98)
                ax.set_ylim([-ylim * 1.5, ylim * 1.5])
                ax.set_ylabel('vel. / nm/s')
