# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 11:43:22 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models.radar            import FrequencyRadar
from fdia_simulation.models.tracks           import Track
from fdia_simulation.attackers.mo_attacker   import MoAttacker
from fdia_simulation.filters.m_radar_filter  import MultipleFreqRadarsFilter
from fdia_simulation.filters.radar_filter_ca import RadarFilterCA


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
    # ====================== Radar filter generation ===========================
    # Filter: constant velocity
    radars = [radar1,radar2]
    radar_filter_ca = MultipleFreqRadarsFilter(dim_x = 9, dim_z = 6, q = 400.,
                                               radars = radars, model = RadarFilterCA,
                                               x0 = 100, y0=100)
    est_xs_ca, est_ys_ca, est_zs_ca = [],[],[]
    for label_val in label_measurements:
        radar_filter_ca.predict()
        radar_filter_ca.update(label_val)
        est_xs_ca.append(radar_filter_ca.x[0,0])
        est_ys_ca.append(radar_filter_ca.x[3,0])
        est_zs_ca.append(radar_filter_ca.x[6,0])

    # for i in range(len(radar_filter_ca.Hs)):
    #     print('H[{0}] = \n{1}\n'.format(i,radar_filter_ca.Hs[i]))
    #     print('Z[{0}] = \n{1}\n\n'.format(i,radar_filter_ca.Zs[i]))
    #     print('===========================================\n\n')
    # ==========================================================================
    # =============================== Plotting =================================

    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='Real position',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad1, ys_from_rad1, zs_from_rad1,color='b',marker='o',alpha = 0.3, label = 'Radar1 measurements')
    ax.scatter(xs_from_rad2, ys_from_rad2, zs_from_rad2,color='m',marker='o',alpha = 0.3, label = 'Radar2 measurements')
    ax.plot(est_xs_ca, est_ys_ca, est_zs_ca,color='orange', label = 'Estimation-CA')
    ax.scatter(radar1.x,radar1.y,radar1.z,color='r', label = 'Radar1', marker = '+')
    ax.scatter(radar2.x,radar2.y,radar2.z,color='g', label = 'Radar2', marker = '+')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
