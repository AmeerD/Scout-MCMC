# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:16:30 2020

@author: ameer
"""

from matplotlib import pyplot as plt
import matplotlib.animation
import numpy as np
import autograd.scipy.stats as ss
from mpl_toolkits import mplot3d


def plotsamples(xlims, ylims, target, samples, init, title, lin=100):
    xvals = np.linspace(xlims[0], xlims[1], lin)
    yvals = np.linspace(ylims[0], ylims[1], lin)
    X, Y = np.meshgrid(xvals, yvals)
    Z = target(np.dstack((X, Y)).reshape((lin * lin, 2))).reshape((lin, lin))

    fig = plt.figure()
    con = fig.add_subplot()
    con.contour(X, Y, Z, linewidths = [0,1.5,1.5,1.5,1.5,1.5,1.5])
    con.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.25)
    con.scatter(init[0], init[1], color="red")
    plt.title(title)
    plt.show()

def plotsamples_3d(samples, known_modes, rotate=True):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    ax.scatter3D(samples[:, 0], samples[:, 1], samples[:, 2]); #, c=zdata, cmap='Greens')
    for i in range(known_modes.shape[0]):
        ax.scatter(known_modes[i,0], known_modes[i,1], known_modes[i,2], color="red", depthshade=True)
        ax.text(known_modes[i,0], known_modes[i,1], known_modes[i,2], "mode", size=12, zorder=1, color="red")
    plt.show()
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    if rotate:
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.show()
            plt.pause(.001)


def animatesamples(xlims, ylims, target, samples, init, title, lin=100):
    xvals = np.linspace(xlims[0], xlims[1], lin)
    yvals = np.linspace(ylims[0], ylims[1], lin)
    X, Y = np.meshgrid(xvals, yvals)
    Z = target(np.dstack((X, Y)).reshape((lin * lin, 2))).reshape((lin, lin))

    fig = plt.figure()
    ax = plt.subplot()

    ax.contour(X, Y, Z)
    ax.scatter(init[0], init[1], color="red")

    def animate(i):
        ax.set_title(title + ', Iteration {}'.format(i))
        if (i > 0) and (not np.any(samples[i, :] == samples[i - 1, :])):
            ax.scatter(samples[i, 0], samples[i, 1], color="blue")

    ani = matplotlib.animation.FuncAnimation(fig, animate, blit=True,
                                             frames=samples.shape[0], repeat=False)

    plt.show()
    return ani


def animatecontours(xlims, ylims, target, samples, covs, init, title, lin=100):
    xvals = np.linspace(xlims[0], xlims[1], lin)
    yvals = np.linspace(ylims[0], ylims[1], lin)
    X, Y = np.meshgrid(xvals, yvals)
    Z = target(np.dstack((X, Y)).reshape((lin * lin, 2))).reshape((lin, lin))

    fig = plt.figure()
    ax = plt.subplot()

    def animate(i):
        ax.cla()
        ax.contour(X, Y, Z, colors='black')
        L = ss.multivariate_normal.pdf(np.dstack((X, Y)), samples[i, :], covs[i, :, :])
        ax.contour(X, Y, L)
        ax.set_title(title + ', Iteration {}'.format(i))

    ani = matplotlib.animation.FuncAnimation(fig, animate, blit=True,
                                             frames=samples.shape[0], repeat=False)

    plt.show()
    return ani
