#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for exercise 2: Determine backazimuth
:copyright:
    Simon Stähler (mail@simonstaehler.com), 2022
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
from scipy.interpolate import interp2d


class Locate2(widgets.HBox):
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

        for event in c.origin_time.keys():
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
        initial_baz = 0.
        self.tP = initial_tP
        self.tS = initial_tS
        self.baz = initial_baz
        self.plot_spec = False
        self.dist = None

        with output:
            self.initialize_figure(initial_tP, initial_tS)
            self.h_dotS = None
            self.h_dotP = None
            self.h_line = None
            self.h_line_cont = None

            self.l_hodo = None
            self.l_seis_E_zoom = None
            self.l_seis_N_zoom = None

            self.set_event(initial_event)

            self.l_seis = dict()
            self.h_spec = dict()
            for tr, ax, comp in zip(self.st, (self.ax_Z, self.ax_N, self.ax_E),
                                    ('Z', 'N', 'E')):
                self.l_seis[comp] = ax.plot(tr.times(), tr.data * 1e9,
                                            lw=c.lw_seis, c=c.color_seis)
                ax.set_xlim(100, 900)

                ax.set_ylabel('vel. / nm/s')
                self.h_spec[comp] = None

            self.equi_lat = None
            self.equi_lon = None
            self.load_equidist()

        # create some control elements
        tP_slider = widgets.IntSlider(value=initial_tP,
                                      min=0, max=1200,
                                      step=1, description='P-arrival')
        tS_slider = widgets.IntSlider(value=initial_tS,
                                      min=0, max=1200,
                                      step=1, description='S-arrival')
        link_P_S = widgets.widget_link.Link(
            (tP_slider, 'value'),
            (tS_slider, 'min'))
        BAZ_slider = widgets.IntSlider(value=initial_baz,
                                       min=0, max=360,
                                       step=1, description='Backazimuth')

        save_button = widgets.Button(description='save event')
        event_combobox = widgets.Dropdown(
            value=initial_event,
            options=c.event_list,
            description='Event'
        )

        tP_slider.observe(self.update_tP, 'value')
        tS_slider.observe(self.update_tS, 'value')
        BAZ_slider.observe(self.update_baz, 'value')
        event_combobox.observe(self.update_event, 'value')
        save_button.on_click(self.save_event)

        controls_1 = widgets.HBox([tP_slider, tS_slider])

        controls_2 = widgets.HBox([event_combobox, BAZ_slider, save_button])
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

        # Plot axis for seismograms and spectrograms
        self.ax_Z = self.fig.add_subplot(gs[0, 0:3])
        self.ax_N = self.fig.add_subplot(gs[1, 0:3], sharex=self.ax_Z)
        self.ax_E = self.fig.add_subplot(gs[2, 0:3], sharex=self.ax_Z)
        self.ax_Z.text(0.97, 0.02, 'vertical', ha='right', va='bottom',
                       transform=self.ax_Z.transAxes, bbox=dict(facecolor='white', alpha=0.9))
        self.ax_N.text(0.97, 0.02, 'north/south', ha='right', va='bottom',
                       transform=self.ax_N.transAxes, bbox=dict(facecolor='white', alpha=0.9))
        self.ax_E.text(0.97, 0.02, 'east/west', ha='right', va='bottom',
                       transform=self.ax_E.transAxes, bbox=dict(facecolor='white', alpha=0.9))
        self.ax_Z.set_xlim(50, 200)
        self.ax_E.set_xlabel('time / second')

        self.l_P_Z = self.ax_Z.axvline(initial_tP, c=c.color_phases['P'])
        self.l_P_N = self.ax_N.axvline(initial_tP, c=c.color_phases['P'], ls='dashed')
        self.l_P_E = self.ax_E.axvline(initial_tP, c=c.color_phases['P'], ls='dashed')
        self.l_S_Z = self.ax_Z.axvline(initial_tS, c=c.color_phases['S'], ls='dashed')
        self.l_S_N = self.ax_N.axvline(initial_tS, c=c.color_phases['S'])
        self.l_S_E = self.ax_E.axvline(initial_tS, c=c.color_phases['S'])

        # Plot axis for hodogram
        self.ax_hodo = self.fig.add_subplot(gs[0:2, 3:5])
        self.ax_hodo.xaxis.tick_top()
        self.ax_hodo.xaxis.set_label_position('top')
        self.ax_hodo.yaxis.tick_right()
        self.ax_hodo.yaxis.set_label_position('right')
        self.ax_hodo.axis('equal')
        self.ax_hodo.set_xlabel('Seismogram, East/West')
        self.ax_hodo.set_ylabel('Seismogram, North/South')
        x = np.sin(np.deg2rad(self.baz)) * 1.
        y = np.cos(np.deg2rad(self.baz)) * 1.
        self.l_baz = self.ax_hodo.plot([0, x], [0, y], lw=1, c='darkgrey', ls='dashed')
        self.h_text = self.ax_hodo.text(0.02, 0.02, 'No distance and baz yet', ha='left', va='bottom',
                                        transform=self.ax_hodo.transAxes)

        # Plot axis for zoomed P-wave
        self.ax_N_zoom = self.fig.add_subplot(gs[2, 3])
        self.ax_E_zoom = self.fig.add_subplot(gs[2, 4], sharex=self.ax_N_zoom, sharey=self.ax_N_zoom)
        self.ax_N_zoom.text(0.97, 0.02, 'north/south', ha='right', va='bottom', size=9,
                            transform=self.ax_N_zoom.transAxes, bbox=dict(facecolor='white', alpha=0.9))
        self.ax_E_zoom.text(0.97, 0.02, 'east/west', ha='right', va='bottom', size=9,
                            transform=self.ax_E_zoom.transAxes, bbox=dict(facecolor='white', alpha=0.9))
        for ax in (self.ax_E_zoom, self.ax_N_zoom):
            ax.axvline(0, ls='dashed', c=c.color_phases['P'])
            ax.set_xlabel('time after P')
            ax.set_yticks([])

        # Plot axis for map with distance
        self.ax_map = self.fig.add_subplot(gs[3:, :])
        self.ax_map.set_ylim(-90., 90.)
        self.ax_map.set_ylabel('latitude')
        self.ax_map.set_xlabel('longitude')
        self.ax_map.set_aspect('equal', 'box')
        self.ax_map.yaxis.tick_right()
        self.ax_map.yaxis.set_label_position('right')
        self.h_circ = None
        self.h_event = None
        self.h_event_saved = []

        # Mark InSight location
        from matplotlib.patches import RegularPolygon
        regpol = RegularPolygon(xy=(c.lon_insight, c.lat_insight), numVertices=3, radius=4,
                                fc='darkred', ec='k')
        self.ax_map.add_patch(regpol)

        img = mpimg.imread('./helpers/MOLA_rolled.png')
        self.ax_map.imshow(img, extent=(-0, 360, -90, 90), cmap='gist_earth')
        self.ax_map.set_xlim(060., 260.)

        # # Plot axis for distance text
        # self.ax_dist_text = self.fig.add_subplot(gs[2, 3:])
        # self.h_text = self.ax_dist_text.text(0.02, 0.97, 'No distance yet', ha='left', va='top',
        #                                      transform=self.ax_dist_text.transAxes)
        # self.ax_dist_text.set_xticks([])
        # self.ax_dist_text.set_yticks([])

    # callback functions
    def update_distance(self, tP, tS):  # , h_dotP, h_dotS, h_line):
        # global h_dotP, h_dotS, h_line, h_line_cont, h_circ
        if self.h_circ is not None:
            # print(self.h_circ)
            # self.h_circ.remove()
            [l.remove() for l in self.h_circ if l is not None]
            self.h_circ = None

        if self.h_event is not None:
            try:
                self.h_event.remove()
                self.h_event = None
            except:
                pass

        self.dist = taup_distance.get_dist(model=self.model, tSmP=tS - tP, depth=50)

        if self.dist is not None:
            # circ = Circle(xy=(c.lon_insight, c.lat_insight), radius=self.dist, ec='k', ls='dashed', fill=False)
            # self.h_circ = self.ax_map.add_patch(circ)

            self.ax_map.set_xlim(060., 260.)
            self.ax_map.yaxis.set_label_position('right')

            azi = np.arange(0, 360, 1)
            self.h_circ = self.ax_map.plot(self.equi_lon(azi, self.dist),
                                           self.equi_lat(azi, self.dist),
                                           c='k', lw=1, ls='dashed')

            lat_event, lon_event = shoot(latitude_1_degree=c.lat_insight,
                                         longitude_1_degree=c.lon_insight,
                                         bearing_degree=self.baz,
                                         distance_km=np.deg2rad(self.dist), radius_km=1.)
            lon_event = lon_event % 360

            circ = Circle(xy=(lon_event, lat_event), radius=2., ec='k', fill=True)
            self.h_event = self.ax_map.add_patch(circ)

            dist_string = f'Distance found!\n' + \
                          f'$t_S - t_P=${tS - tP:5.1f} sec\n' + \
                          f'Distance: {self.dist:5.1f}° \n' + \
                          f'Backazimuth : {self.baz:5.1f}°\n' + \
                          f'Event pos.: {lon_event:5.1f}°E, {lat_event:5.1f}°N '
            self.h_text.set_text(dist_string)
            self.h_text.set_color('black')
            self.h_text.set_weight('bold')

        else:
            self.h_dotS = None
            self.h_dotP = None
            self.h_line = None
            self.h_line_cont = None
            '''
            dist_string = f'No Distance found\n' + \
                          f'for $t_S - t_P=${tS - tP:5.1f} sec\n'
            self.h_text.set_text(dist_string)
            self.h_text.set_color('darkred')
            self.h_text.set_weight('bold')
            '''

    def update_tP(self, change):
        # global tP, tS
        """redraw line (update plot)"""
        self.l_P_Z.set_xdata([change.new, change.new])
        self.l_P_N.set_xdata([change.new, change.new])
        self.l_P_E.set_xdata([change.new, change.new])
        self.update_distance(tP=change.new, tS=self.tS)
        self.fig.canvas.draw()
        self.tP = change.new
        self.update_hodogram_and_zoom()

    def update_tS(self, change):
        # global tP, tS
        """redraw line (update plot)"""
        self.l_S_Z.set_xdata([change.new, change.new])
        self.l_S_E.set_xdata([change.new, change.new])
        self.l_S_N.set_xdata([change.new, change.new])
        self.update_distance(tP=self.tP, tS=change.new)
        self.fig.canvas.draw()
        self.tS = change.new

    def update_event(self, change):
        self.set_event(change.new)

        for comp, ax in zip(['Z', 'N', 'E'], (self.ax_Z, self.ax_N, self.ax_E)):
            for tr, comp, ax in zip(self.st, ['Z', 'N', 'E'], (self.ax_Z, self.ax_N, self.ax_E)):
                [l.remove() for l in self.l_seis[comp] if l is not None]
                ax.set_yscale('linear')
                self.l_seis[comp] = ax.plot(tr.times(), tr.data * 1e9,
                                            lw=c.lw_seis, c=c.color_seis)
                ylim = np.percentile(abs(tr.data * 1e9), q=98)
                ax.set_ylim([-ylim * 1.5, ylim * 1.5])
                ax.set_ylabel('vel. / nm/s')
        self.update_hodogram_and_zoom()

    def update_hodogram_and_zoom(self):
        # Update Hodogram
        st_work = self.st.copy()
        tr = st_work[0]
        st_work.filter('highpass', freq=0.2)
        st_work.filter('lowpass', freq=0.9)
        st_work = st_work.slice(starttime=tr.stats.starttime + self.tP - 5,
                                endtime=tr.stats.starttime + self.tP + 10)
        tr_N = st_work.select(channel='BHN')[0]
        tr_E = st_work.select(channel='BHE')[0]
        if self.l_hodo is not None:
            [l.remove() for l in self.l_hodo]
        self.l_hodo = self.ax_hodo.plot(tr_E.data * 1e9, tr_N.data * 1e9, c=c.color_seis, lw=c.lw_seis)

        if self.l_seis_N_zoom is not None:
            [l.remove() for l in self.l_seis_N_zoom]
        self.l_seis_N_zoom = self.ax_N_zoom.plot(np.linspace(-5., 10., tr_N.stats.npts),
                                                 tr_N.data * 1e9, lw=c.lw_seis, c=c.color_seis)

        if self.l_seis_E_zoom is not None:
            [l.remove() for l in self.l_seis_E_zoom]
        self.l_seis_E_zoom = self.ax_E_zoom.plot(np.linspace(-5., 10., tr_E.stats.npts),
                                                 tr_E.data * 1e9, lw=c.lw_seis, c=c.color_seis)

    def save_event(self, change):
        print(self.event, self.baz, self.dist)
        if self.dist is not None:
            lat_event, lon_event = shoot(latitude_1_degree=c.lat_insight, longitude_1_degree=c.lon_insight,
                                         bearing_degree=self.baz, distance_km=np.deg2rad(self.dist), radius_km=1.)
            lon_event = lon_event % 360
            circ = Circle(xy=(lon_event, lat_event), radius=2., ec='k', fill=True, label=self.event)
            self.h_event_saved.append(self.ax_map.add_patch(circ))
            self.ax_map.legend()

    def update_baz(self, change):
        self.baz = change.new
        length = abs(self.ax_hodo.get_ylim()[1])
        x = np.sin(np.deg2rad(self.baz)) * length / 2.
        y = np.cos(np.deg2rad(self.baz)) * length / 2.
        if self.l_baz is not None:
            [l.remove() for l in self.l_baz]
        self.l_baz = self.ax_hodo.plot([0, x], [0, y], lw=c.lw_baz, c=c.color_baz, ls='dashed')

        if self.dist is not None:
            lat_event, lon_event = shoot(latitude_1_degree=c.lat_insight, longitude_1_degree=c.lon_insight,
                                         bearing_degree=self.baz, distance_km=np.deg2rad(self.dist), radius_km=1.)
            lon_event = lon_event % 360
            circ = Circle(xy=(lon_event, lat_event), radius=2., ec='k', fill=True)
            if self.h_event is not None:
                try:
                    self.h_event.remove()
                except:
                    pass
            self.h_event = self.ax_map.add_patch(circ)

            dist_string = f'Distance found!\n' + \
                          f'$t_S - t_P=${self.tS - self.tP:5.1f} sec\n' + \
                          f'Distance: {self.dist:5.1f}° \n' + \
                          f'Backazimuth : {self.baz:5.1f}°\n' + \
                          f'Event pos.: {lon_event:5.1f}°E, {lat_event:5.1f}°N '
            self.h_text.set_text(dist_string)
            self.h_text.set_color('black')
            self.h_text.set_weight('bold')

    def load_equidist(self):
        dat = np.load('helpers/equidistances.npz')
        self.equi_lat = interp2d(x=dat['azis'], y=dat['dists'], z=dat['lats'])
        self.equi_lon = interp2d(x=dat['azis'], y=dat['dists'], z=dat['lons'])


def shoot(latitude_1_degree, longitude_1_degree, bearing_degree, distance_km, radius_km):
    """
    Shoot a ray from point in direction for certain length and return where you land
    (Direct geodetic problem). Works on sphere
    :param latitude_1_degree: latitude of starting point
    :param longitude_1_degree: longitude of starting point
    :param bearing_degree: bearing from north, CW
    :param distance_km: distance in kilometer
    :param radius_km: radius of planet
    :return: latitude, longitude of target
    """
    lat1 = np.deg2rad(latitude_1_degree)
    lon1 = np.deg2rad(longitude_1_degree)
    bearing = np.deg2rad(bearing_degree)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance_km / radius_km) +
                     np.cos(lat1) * np.sin(distance_km / radius_km) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance_km / radius_km) * np.cos(lat1),
                             np.cos(distance_km / radius_km) - np.sin(lat1) * np.sin(lat2))
    return np.rad2deg(lat2), np.mod(np.rad2deg(lon2) + 540., 360.) - 180.
