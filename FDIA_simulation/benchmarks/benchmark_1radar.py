# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 14:05:39 2019

@author: qde
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from fdia_simulation.models.tracks           import Track
from fdia_simulation.models.radar            import Radar
from fdia_simulation.filters.radar_filter_cv import RadarFilterCV
from fdia_simulation.filters.radar_filter_ca import RadarFilterCA
from fdia_simulation.filters.radar_filter_ta import RadarFilterTA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",choices=["cv","ca","ta"],
                        help="Model used by the estimator (cv,ca or ta)")
    parser.add_argument("-t","--track",
                        help="Trajectory of the modeled airplane")
    args = parser.parse_args()

    trajectory = Track()
    xs, ys, zs = [], [], []

    methodName = getattr(trajectory,"gen_"+args.track)
    xs, ys, zs = methodName()
    position_data = np.array(list(zip(xs,ys,zs)))

    radar = Radar(x=800,y=800)
    rs, thetas, phis = radar.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
    xs_from_rad, ys_from_rad, zs_from_rad = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)
    radar_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    radar_computed_values = np.array(list(zip(xs_from_rad, ys_from_rad, zs_from_rad)))

    class_name = "RadarFilter" + args.model.upper()
    filter_args = {"dim_x" : 9, "dim_z" : 3, "q" : 400.,
                   "x0" : 100, "y0" : 100, "z0":8000, "x_rad":800, "y_rad":800}
    filter_class = globals()[class_name]
    filter = filter_class(**filter_args)

    est_xs_ca, est_ys_ca, est_zs_ca = [],[],[]
    for val in radar_values:
        filter.predict()
        filter.update(val)
        est_xs_ca.append(filter.x[0,0])
        est_ys_ca.append(filter.x[3,0])
        est_zs_ca.append(filter.x[6,0])

    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='plot test',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad, ys_from_rad, zs_from_rad,color='b',marker='o',alpha = 0.3, label = 'Radar measurements')
    ax.plot(est_xs_ca, est_ys_ca, est_zs_ca,color='m', label = args.model)
    ax.scatter(radar.x,radar.y,radar.z,color='r', label = 'Radar')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
