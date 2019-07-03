# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:50:21 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from numpy import dot
from fdia_simulation.models.moving_target          import Command
from fdia_simulation.models.maneuvered_aircraft    import ManeuveredAircraft
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.attackers.mo_attacker         import MoAttacker
from fdia_simulation.filters.radar_filter_ca       import RadarFilterCA
from fdia_simulation.models.tracks                 import Track


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    xs, ys, zs = trajectory.gen_takeoff()
    position_data = np.array(list(zip(xs,ys,zs)))
    # ==========================================================================
    # ======================== Radar data generation ===========================
    radar = Radar(x=800,y=800)
    rs, thetas, phis = radar.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
    xs_from_rad, ys_from_rad, zs_from_rad = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)

    radar_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    radar_computed_values = np.array(list(zip(xs_from_rad, ys_from_rad, zs_from_rad)))
    # print("Noisy radar values: \n{0}\n".format(radar_values[:10]))
    # print("Radar computed position values: \n{0}\n".format(radar_computed_values[:10]))
    # # ==========================================================================
    # # ====================== Radar filter generation ===========================
    # Filter: constant acceleration
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)
    est_xs_ca, est_ys_ca, est_zs_ca = [],[],[]
    for val in radar_values:
        radar_filter_ca.predict()
        radar_filter_ca.update(val)
        est_xs_ca.append(radar_filter_ca.x[0,0])
        est_ys_ca.append(radar_filter_ca.x[3,0])
        est_zs_ca.append(radar_filter_ca.x[6,0])
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='Real position',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad, ys_from_rad, zs_from_rad,color='b',marker='o',alpha = 0.3, label = 'Radar measurements')
    ax.plot(est_xs_ca, est_ys_ca, est_zs_ca,color='m', label = 'Estimation-CA')
    ax.scatter(radar.x,radar.y,radar.z,color='r', label = 'Radar')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
