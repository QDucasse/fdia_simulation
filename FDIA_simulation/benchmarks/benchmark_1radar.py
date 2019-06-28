# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:41:38 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from numpy import dot
from filterpy.kalman                               import KalmanFilter, ExtendedKalmanFilter, IMMEstimator
from fdia_simulation.models.moving_target          import Command
from fdia_simulation.models.maneuvered_aircraft    import ManeuveredAircraft
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.attackers.mo_attacker         import MoAttacker
from fdia_simulation.filters.radar_filter          import RadarFilterCV, RadarFilterCA



if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    # Route generation example with a ManeuveredAircraft
    sensor_std = 1.
    headx_cmd = Command('headx',0,0,0)
    headz_cmd = Command('headz',0,0,0)
    vel_cmd   = Command('vel',1,0,0)
    aircraft  = ManeuveredAircraft(x0 = 1000, y0 = 1000, z0=1, v0 = 0, hx0 = 0, hz0 = 0, command_list = [headx_cmd, headz_cmd, vel_cmd])
    xs, ys, zs = [], [], []

    # Take off acceleration objective
    aircraft.change_command("vel",200, 20)
    # First phase -> Acceleration
    for i in range(10):
        x, y, z = aircraft.update()
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Change in commands -> Take off
    aircraft.change_command("headx",315, 25)
    aircraft.change_command("headz",315, 25)

    # Second phase -> Take off
    for i in range(30):
        x, y, z = aircraft.update()
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Change in commands -> Steady state
    aircraft.change_command("headx",90, 25)
    aircraft.change_command("headz",270, 25)

    # Third phase -> Steady state
    for i in range(60):
        x, y, z = aircraft.update()
        xs.append(x)
        ys.append(y)
        zs.append(z)

    position_data = np.array(list(zip(xs,ys,zs)))
    # ==========================================================================
    # ======================== Radar data generation ===========================
    # Radar 1
    radar = Radar(x=800,y=800)
    rs, thetas, phis = radar.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
    xs_from_rad1, ys_from_rad1, zs_from_rad1 = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)

    radar_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    # print("Noisy radar values: \n{0}\n".format(radar_values[:10]))
    radar_computed_values = np.array(list(zip(xs_from_rad1, ys_from_rad1, zs_from_rad1)))
    # print("Radar computed position values: \n{0}\n".format(radar_computed_values[:10]))
    # # ==========================================================================
    # # ====================== Radar filter generation ===========================
    # Filter n°1: constant velocity
    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3,x0 = 500,y0=500, x_rad=800,y_rad=800)
    est_xs_cv, est_ys_cv, est_zs_cv = [],[],[]
    for val in radar_values:
        radar_filter_cv.predict()
        radar_filter_cv.update(val)
        est_xs_cv.append(radar_filter_cv.x[0,0])
        est_ys_cv.append(radar_filter_cv.x[1,0])
        est_zs_cv.append(radar_filter_cv.x[2,0])

    radar_estimated_values = np.array(list(zip(est_xs_cv,est_ys_cv,est_zs_cv)))
    # print("Estimated radar values: \n{0}\n".format(radar_estimated_values[:10]))

    # Filter n°2: constant acceleration
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3,
                                    x0 = 500, y0 = 500,  z0 = 0,
                                    vx0 = 0,  vy0 = 100, vz0 = 0,
                                    ax0 = 0,  ay0 = 0,   az0 = 0,
                                    x_rad=800,y_rad=800)
    est_xs_ca, est_ys_ca, est_zs_ca = [],[],[]
    for val in radar_values:
        radar_filter_ca.predict()
        radar_filter_ca.update(val)
        est_xs_ca.append(radar_filter_ca.x[0,0])
        est_ys_ca.append(radar_filter_ca.x[1,0])
        est_zs_ca.append(radar_filter_ca.x[2,0])
    # ==========================================================================
    # ========================= IMM generation =================================
    # radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3,x0 = 500,y0=500, x_rad=800,y_rad=800)
    # radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3,x0 = 500,y0=500, x_rad=800,y_rad=800)
    # filters = [radar_filter_cv, radar_filter_ca]
    # mu = [0.5, 0.5]
    # trans = np.array([[0.97, 0.03], [0.03, 0.97]])
    # imm = IMMEstimator(filters, mu, trans)
    #
    # est_xs_imm, est_ys_imm, est_zs_imm = [],[],[]
    # probs = []
    # for val in radar_values:
    #     imm.predict()
    #     imm.update(val)
    #     est_xs_imm.append(imm.x[0,0])
    #     est_ys_imm.append(imm.x[1,0])
    #     est_zs_imm.append(imm.x[2,0])
    #     probs.append(imm.mu)
    #
    # probs = np.array(probs)
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='plot test',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad1, ys_from_rad1, zs_from_rad1,color='b',marker='o',alpha = 0.3, label = 'Radar measurements')
    ax.plot(est_xs_ca, est_ys_ca, est_zs_ca,color='m', label = 'Constant accleration')
    ax.plot(est_xs_cv, est_ys_cv, est_zs_cv,color='orange', label = 'Constant velocity')
    # ax.plot(est_xs_imm, est_ys_imm, est_zs_imm,color='orange', label = 'IMM')
    ax.scatter(radar.x,radar.y,radar.z,color='r', label = 'Radar')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    fig.show()

    # fig2 = plt.figure(2)
    # plt.plot(probs[:,0])
    # plt.plot(probs[:,1])
    # fig2.show()
    plt.show()
