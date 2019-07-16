# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 11:47:28 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from filterpy.kalman import IMMEstimator
from fdia_simulation.models.radar            import FrequencyRadar
from fdia_simulation.models.tracks           import Track
from fdia_simulation.filters.radar_filter_cv import RadarFilterCV
from fdia_simulation.filters.radar_filter_ca import RadarFilterCA
from fdia_simulation.filters.radar_filter_ct import RadarFilterCT
from fdia_simulation.filters.m_radar_filter  import MultipleFreqRadarsFilter

if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_takeoff()
    xs, ys, zs = states[:,0], states[:,3], states[:,6]
    position_data = np.array(list(zip(xs,ys,zs)))

    dt_rad1 = 0.1
    dt_rad2 = 0.4
    # ==========================================================================
    # ======================== Radars data generation ===========================
    # Radar 1: Precision radar
    radar1 = FrequencyRadar(tag = 0, x = 800, y = 800, dt = dt_rad1)
    rs1, thetas1, phis1 = radar1.gen_data(position_data)
    noisy_rs1, noisy_thetas1, noisy_phis1 = radar1.sense(rs1, thetas1, phis1)
    xs_from_rad1, ys_from_rad1, zs_from_rad1 = radar1.radar2cartesian(noisy_rs1, noisy_thetas1, noisy_phis1)

    radar1_values = np.array(list(zip(noisy_rs1, noisy_thetas1, noisy_phis1)))
    # print("Noisy radar1 values: \n{0}\n".format(radar1_values[:10]))
    radar1_computed_values = np.array(list(zip(xs_from_rad1, ys_from_rad1, zs_from_rad1)))
    # print("radar1 computed position values: \n{0}\n".format(radar1_computed_values[:10]))
    label_measurements1 = radar1.compute_measurements(position_data)
    # print("radar2 computed measurements info: \n{0}\n".format(measurements_info2[:10]))

    # Radar 2: Standard radar
    radar2 = FrequencyRadar(tag = 1, x = 800, y = 200, dt = dt_rad2,
                            r_noise_std = 5., theta_noise_std = 0.005, phi_noise_std = 0.005)
    rs2, thetas2, phis2 = radar2.gen_data(position_data)
    noisy_rs2, noisy_thetas2, noisy_phis2 = radar2.sense(rs2, thetas2, phis2)
    xs_from_rad2, ys_from_rad2, zs_from_rad2 = radar2.radar2cartesian(noisy_rs2, noisy_thetas2, noisy_phis2)

    radar2_values = np.array(list(zip(noisy_rs2, noisy_thetas2, noisy_phis2)))
    # print("Noisy radar2 values: \n{0}\n".format(radar2_values[:10]))
    radar2_computed_values = np.array(list(zip(xs_from_rad2, ys_from_rad2, zs_from_rad2)))
    # print("radar2 computed position values: \n{0}\n".format(radar2_computed_values[:10]))
    label_measurements2    = radar2.compute_measurements(position_data)
    # print("radar2 computed measurements info: \n{0}\n".format(measurements_info2[:10]))

    # Combination of the measurements
    label_measurements    = sorted(label_measurements1 + label_measurements2)
    # ==========================================================================
    # ========================= IMM generation =================================
    radars = [radar1,radar2]
    radar_filter_cv = MultipleFreqRadarsFilter(dim_x = 9, dim_z = 6, q = 1.,
                                               radars = radars, model = RadarFilterCV,
                                               x0 = 100, y0=100)
    radar_filter_ca = MultipleFreqRadarsFilter(dim_x = 9, dim_z = 6, q = 400.,
                                               radars = radars, model = RadarFilterCA,
                                               x0 = 100, y0=100)
    radar_filter_ct = MultipleFreqRadarsFilter(dim_x = 9, dim_z = 6, q = 350.,
                                               radars = radars, model = RadarFilterCT,
                                               x0 = 100, y0=100)
    filters = [radar_filter_cv, radar_filter_ca, radar_filter_ct]
    mu = [0.33, 0.33, 0.33]
    trans = np.array([[0.998, 0.001, 0.001],
                      [0.050, 0.900, 0.050],
                      [0.001, 0.001, 0.998]])
    imm = IMMEstimator(filters, mu, trans)

    est_xs_imm, est_ys_imm, est_zs_imm = [],[],[]
    probs = []
    for label_val in label_measurements:
        imm.predict()
        imm.update(label_val)
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
    ax.scatter(xs_from_rad1, ys_from_rad1, zs_from_rad1,color='b',marker='o',alpha = 0.3, label = 'Radar1 measurements')
    ax.scatter(xs_from_rad2, ys_from_rad2, zs_from_rad2,color='m',marker='o',alpha = 0.3, label = 'Radar2 measurements')
    ax.plot(est_xs_imm, est_ys_imm, est_zs_imm,color='orange', label='Estimation-IMM3')
    ax.scatter(radar1.x,radar1.y,radar1.z,color='r', label = 'Radar1')
    ax.scatter(radar2.x,radar2.y,radar2.z,color='g', label = 'Radar2')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    fig.show()


    fig2 = plt.figure(2)
    plt.plot(probs[:,0],label='Constant Velocity')
    plt.plot(probs[:,1],label='Constant Acceleration')
    plt.plot(probs[:,2],label='Constant Turn')
    plt.legend()
    fig2.show()
    plt.show()
