#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2020
:license:
    None
"""
from argparse import ArgumentParser
from collections import OrderedDict
from os.path import split as psplit, splitext as psplitext

import numpy as np
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel
from obspy.taup.helper_classes import TauModelError
from obspy.taup.taup_create import build_taup_model


def define_arguments():
    helptext = "Compute taup_distance from TauP model, given a S-P " \
               "traveltime difference"
    parser = ArgumentParser(description=helptext)

    helptext = "Input TauP file"
    parser.add_argument("fnam_in", help=helptext)

    helptext = "S-P arrival time differences"
    parser.add_argument("times", nargs="+", type=float, help=helptext)

    helptext = "Plot rays of best solution"
    parser.add_argument("--plot_rays", action="store_true", default=False,
                        help=helptext)

    helptext = "Plot convergence in T-X plot (default: False)"
    parser.add_argument("--plot", action="store_true", default=False,
                        help=helptext)
    return parser.parse_args()


def check_nd_for_innercore(fnam_nd_in,
                           fnam_nd_out):
    """
    Check whether an nd file has an inner core. Many people create "lazy" models for Mars, without an inner core, but
    TauP is extremely geocentric and always assumes a planet with crust, mantle, outer and inner core.
    Returns True if the model already had an inner core. If not, creates a file "fnam_nd_out" with an IC between the
    last two layers.
    :param fnam_nd_in: file name to test
    :param fnam_nd_out: file name of model file with added inner core
    :return: True if the model already had an inner core
    """
    import re
    pattern = "inner-core"
    has_inner_core = False
    nlines = 0
    with open(fnam_nd_in, "r") as f:
        for line in f:
            nlines += 1
            if re.search(pattern, line):
                has_inner_core = True

    if not has_inner_core:
        with open(fnam_nd_in, "r") as f_in, open(fnam_nd_out, "w") as f_out:
            for iline, line in enumerate(f_in):
                f_out.write(line)
                if iline == nlines - 2:
                    f_out.write(pattern + "\n")
                    f_out.write(line)
    return has_inner_core


def deck2nd(fnam_deck: str,
            fnam_nd: str):
    """
    Convert a MINEOS deck file to TauP nd (named discontinuities).
    :param fnam_deck: file name of .deck file (in)
    :param fnam_nd: file name of nd file (out)
    :return:
    """
    model_deck = np.loadtxt(fname=fnam_deck, skiprows=3,
                            usecols=(0, 1, 2, 3))
    with open(file=fnam_deck, mode="r") as f:
        _ = f.readline()
        _ = f.readline()
        line = f.readline()
        nlines, nic, noc, tmp1, tmp2, nmoho, nconrad = line.split()
    nlines = int(nlines) - 1
    nic = int(nic)
    noc = int(noc)
    nmoho = int(nmoho)
    radius_planet = model_deck[-1][0] * 1e-3

    with open(file=fnam_nd, mode="w") as f_nd:
        for iline in range(nlines, 0, -1):
            depth = radius_planet - model_deck[iline][0] * 1e-3
            vp = model_deck[iline][2] * 1e-3
            vs = model_deck[iline][3] * 1e-3
            # Some people like to create quasi-liquid layers
            if vs < 0.1:
                vs = 0.
            rho = model_deck[iline][1] * 1e-3
            f_nd.write("%10.4f %7.4f %7.4f %7.4f\n" % (depth, vp, vs, rho))
            if iline == nic:
                f_nd.write("inner-core\n")
            elif iline == noc:
                f_nd.write("outer-core\n")
            elif iline == nmoho:
                f_nd.write("mantle\n")


def get_dist(model: TauPyModel,
             tSmP: float,
             depth: float,
             phase_list=("P", "S"),
             plot_convergence=False):
    """
    Get taup_distance of an event, given difference between
    two phase arrival times (default: P and S)
    :param model: TauPyModel object for given velocity model
    :param tSmP: time difference between arrivals in seconds
    :param depth: depth of event in km
    :param phase_list: list-like object with two phase names
    :param plot_convergence: plot convergence (default: False)
    :return: taup_distance of event in degree
    """
    from scipy.optimize import newton

    # Reasonable guess 1
    dist0 = tSmP / 6.5
    try:
        dist = newton(func=_get_TSmP, fprime=_get_SSmP,
                      x0=dist0, args=(model, tSmP, phase_list,
                                      plot_convergence, depth),
                      maxiter=10)
    except RuntimeError:
        dist = None
    if dist is None:
        # Reasonable guess 2
        dist0 = tSmP / 8.
        try:
            dist = newton(func=_get_TSmP, fprime=_get_SSmP,
                          x0=dist0,
                          args=(model, tSmP, phase_list,
                                plot_convergence, depth),
                          maxiter=10)
        except RuntimeError:
            dist = None

    if plot_convergence:
        plt.axhline(tSmP)
        plt.show()
    return dist


def _get_TSmP(distance: float,
              model: TauPyModel,
              tmeas: float,
              phase_list: tuple,
              plot: bool,
              depth: float):
    """
    Compute travel time difference between two phases (minus tmeas)
    :param distance: taup_distance in degree
    :param model: TauPy model
    :param tmeas: measured travel time difference
    :param phase_list: phase names
    :param plot: plot convergence
    :param depth: depth of event in km
    :return: travel time difference between phases minus tmeas
    """
    if len(phase_list) != 2:
        raise ValueError("Only two phases allowed")
    tP = None
    tS = None
    try:
        arrivals = model.get_travel_times(source_depth_in_km=depth,
                                          distance_in_degree=distance,
                                          phase_list=phase_list)
    except (ValueError, TauModelError):
        pass
    else:
        for arr in arrivals:
            if arr.name == phase_list[0] and tP is None:
                tP = arr.time
            elif arr.name == phase_list[1] and tS is None:
                tS = arr.time
    if tP is None or tS is None:
        if plot:
            plt.plot(distance, -1000, "o")
        return -1000.
    else:
        if plot:
            plt.plot(distance, tS - tP, "o")
        return (tS - tP) - tmeas


def _get_SSmP(distance: float,
              model: TauPyModel,
              tmeas: float,
              phase_list: tuple,
              plot: bool,
              depth: float):
    """
    Compute derivative of travel time difference between two phases (minus tmeas)
    Uses slowness of the two phases.
    Used to compute derivative for fitting. Some unused arguments,
    but argument list needs to be identical to _get_TSmP.
    :param distance: taup_distance in degree
    :param model: TauPy model
    :param tmeas: measured travel time difference
    :param phase_list: phase names
    :param plot: plot convergence
    :param depth: depth of event in km
    :return: travel time difference between phases minus tmeas
    """

    if len(phase_list) != 2:
        raise ValueError("Only two phases allowed")
    sP = None
    sS = None
    try:
        arrivals = model.get_travel_times(source_depth_in_km=depth,
                                          distance_in_degree=distance,
                                          phase_list=phase_list)
    except (ValueError, TauModelError):
        pass
    else:
        for arr in arrivals:
            if arr.name == phase_list[0] and sP is None:
                sP = np.deg2rad(arr.ray_param)
            elif arr.name == phase_list[1] and sS is None:
                sS = np.deg2rad(arr.ray_param)

    if sP is None or sS is None:
        return -10000.
    else:
        return sS - sP


def main(fnam_nd: str,
         times: tuple,
         phase_list=("P", "S"),
         depth=40.,
         plot_rays=False):
    """
    Compute distance of an event, given S-P time
    :param fnam_nd: name of TauP compatible model file
    :param times: list of S-P time
    :param phase_list: list of phases between which the time difference is measured (usually P and S)
    :param depth: assumed depth of event
    :param plot_rays: create a plot with the ray paths
    """
    fnam_npz = "./taup_tmp/" \
               + psplit(fnam_nd)[-1][:-3] + ".npz"
    build_taup_model(fnam_nd,
                     output_folder="./taup_tmp"
                     )
    cache = OrderedDict()
    model = TauPyModel(model=fnam_npz, cache=cache)

    if plot_rays:
        fig, ax = plt.subplots(1, 1)

    for itime, tSmP in enumerate(times):
        dist = get_dist(model, tSmP=tSmP, depth=depth, phase_list=phase_list)
        if dist is None:
            print(f"{fnam_nd}, S-P time: {tSmP:5.1f}: NO SOLUTION FOUND!")
        else:
            print(f"{fnam_nd}, S-P time: {tSmP:5.1f}, "
                  f"taup_distance: {dist:5.1f}")

        if plot_rays:
            if dist is None:
                ax.plot((-200), (-200),
                        label="%4.1f sec, NO SOLUTION" % (tSmP), c="white",
                        lw=0.0)
            else:
                arrivals = model.get_ray_paths(distance_in_degree=dist,
                                               source_depth_in_km=depth,
                                               phase_list=["P", "S"])

                RADIUS_MARS = 3389.5
                already_plotted = dict(P=False,
                                       S=False)
                ls = dict(P="solid", S="dashed")
                label = dict(P="%4.1f sec, %5.1f°" % (tSmP, dist),
                             S=None)
                for arr in arrivals:
                    if not already_plotted[arr.name]:
                        already_plotted[arr.name] = True
                        x = (RADIUS_MARS - arr.path["depth"]) * \
                            np.sin(arr.path["dist"])
                        y = (RADIUS_MARS - arr.path["depth"]) * \
                            np.cos(arr.path["dist"])
                        ax.plot(x, y, c="C%d" % itime, ls=ls[arr.name],
                                label=label[arr.name],
                                lw=1.2)

    if plot_rays:
        for layer_depth in model.model.get_branch_depths():  # (0, 10, 50, 1000,
            # 1500):
            angles = np.linspace(0, 2 * np.pi, 1000)
            x_circle = (RADIUS_MARS - layer_depth) * np.sin(angles)
            y_circle = (RADIUS_MARS - layer_depth) * np.cos(angles)
            ax.plot(x_circle, y_circle, c="k", ls="dashed", lw=0.5,
                    zorder=-1)
        for layer_depth in (model.model.cmb_depth, 0.0):
            angles = np.linspace(0, 2 * np.pi, 1000)
            x_circle = (RADIUS_MARS - layer_depth) * np.sin(angles)
            y_circle = (RADIUS_MARS - layer_depth) * np.cos(angles)
            ax.plot(x_circle, y_circle, c="k", ls="solid", lw=1.0)

        #  for layer_depth in [1100.0]:
        #      angles = np.linspace(0, 2 * np.pi, 1000)
        #      x_circle = (RADIUS_MARS - layer_depth) * np.sin(angles)
        #      y_circle = (RADIUS_MARS - layer_depth) * np.cos(angles)
        #      ax.plot(x_circle, y_circle, c="k", ls="dotted", lw=0.3)

        ax.set_xlim(-100, RADIUS_MARS + 100)
        ax.set_ylim(1000, RADIUS_MARS + 100)
        ax.set_xlabel("radius / km")
        ax.set_ylabel("radius / km")
        ax.set_title("Ray path for model %s" % fnam_nd)
        ax.set_aspect("equal", "box")
        ax.legend(loc="lower left")
        plt.show()


if __name__ == "__main__":
    args = define_arguments()
    if args.fnam_in[-5:] == ".deck":
        fnam_nd = psplitext(psplit(args.fnam_in)[-1])[0] + ".nd"
        deck2nd(args.fnam_in, fnam_nd=fnam_nd)
    else:
        fnam_nd = psplitext(psplit(args.fnam_in)[-1])[0] + "_ic.nd"
        had_ic = check_nd_for_innercore(fnam_nd_in=args.fnam_in,
                                        fnam_nd_out=fnam_nd)
        if had_ic:
            fnam_nd = args.fnam_in

    main(fnam_nd=fnam_nd, times=args.times, plot_rays=args.plot_rays)
