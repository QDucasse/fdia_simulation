# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 13:28:20 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from filterpy.kalman import IMMEstimator
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.models.tracks                 import Track
from fdia_simulation.attackers.mo_attacker         import MoAttacker
from fdia_simulation.filters.radar_filter_cv       import RadarFilterCV
from fdia_simulation.filters.radar_filter_ca       import RadarFilterCA
from fdia_simulation.filters.radar_filter_turn     import RadarFilterTurn
from fdia_simulation.filters.radar_filter_ta       import RadarFilterTA



if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    xs, ys, zs = trajectory.gen_takeoff()
    position_data = np.array(list(zip(xs,ys,zs)))
    # ==========================================================================
    # ======================== Radar data generation ===========================
    # Radar 1
    radar = Radar(x=800,y=800)
    rs, thetas, phis = radar.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
    xs_from_rad, ys_from_rad, zs_from_rad = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)

    radar_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    # print("Noisy radar values: \n{0}\n".format(radar_values[:10]))
    radar_computed_values = np.array(list(zip(xs_from_rad, ys_from_rad, zs_from_rad)))
    # print("Radar computed position values: \n{0}\n".format(radar_computed_values[:10]))
    # ==========================================================================
    # ========================= IMM generation =================================
    radar_filter_cv   = RadarFilterCV(dim_x = 9, dim_z = 3, q = 1.,x0 = 100.,y0=100.,radar = radar)
    radar_filter_ca   = RadarFilterCA(dim_x = 9, dim_z = 3, q = 400.,x0 = 100.,y0=100.,radar = radar)
    radar_filter_turn = RadarFilterTurn(dim_x = 9, dim_z = 3, q = 350.,x0 = 100.,y0=100.,radar = radar)
    radar_filter_ta   = RadarFilterTA(dim_x = 9, dim_z = 3, q = 25.,x0 = 100.,y0=100.,radar = radar)
    filters = [radar_filter_cv, radar_filter_ca, radar_filter_turn, radar_filter_ta]
    mu = [0.25, 0.25, 0.25, 0.25]
    trans = np.array([[0.997, 0.001, 0.001, 0.001],
                      [0.050, 0.850, 0.050, 0.050],
                      [0.001, 0.001, 0.001, 0.997],
                      [0.001, 0.001, 0.001, 0.997]])
    imm = IMMEstimator(filters, mu, trans)

    est_xs_imm, est_ys_imm, est_zs_imm = [],[],[]
    probs = []
    for val in radar_values:
        imm.predict()
        imm.update(val)
        est_xs_imm.append(imm.x[0,0])
        est_ys_imm.append(imm.x[3,0])
        est_zs_imm.append(imm.x[6,0])
        probs.append(imm.mu)

    probs = np.array(probs)
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='plot test',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad, ys_from_rad, zs_from_rad,color='b',marker='o',alpha = 0.3, label = 'Radar measurements')
    ax.plot(est_xs_imm, est_ys_imm, est_zs_imm,color='orange', label='Estimation-IMM4')
    ax.scatter(radar.x,radar.y,radar.z,color='r', label = 'Radar')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    fig.show()


    fig2 = plt.figure(2)
    plt.plot(probs[:,0],label='Constant Velocity')
    plt.plot(probs[:,1],label='Constant Acceleration')
    plt.plot(probs[:,2],label='Constant Turn')
    plt.plot(probs[:,3],label='Thrust Acceleration')
    plt.legend()
    fig2.show()
    plt.show()
