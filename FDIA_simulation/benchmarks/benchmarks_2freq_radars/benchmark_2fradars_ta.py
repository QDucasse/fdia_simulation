# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 11:47:27 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models.radar            import FrequencyRadar
from fdia_simulation.models.tracks           import Track
from fdia_simulation.attackers.mo_attacker   import MoAttacker
from fdia_simulation.filters.m_radar_filter  import MultipleFreqRadarsFilter
from fdia_simulation.filters.radar_filter_ta import RadarFilterTA


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    dt_rad1 = 0.1
    trajectory1 = Track(dt = dt_rad1)
    xs1, ys1, zs1 = trajectory1.gen_takeoff()
    position_data1 = np.array(list(zip(xs1,ys1,zs1)))

    dt_rad2 = 0.4
    trajectory2 = Track(dt = dt_rad2)
    xs2, ys2, zs2 = trajectory2.gen_takeoff()
    position_data2 = np.array(list(zip(xs2,ys2,zs2)))
    # ==========================================================================
    # ======================== Radars data generation ===========================
    # Radar 1: Precision radar
    radar1 = FrequencyRadar(tag = 0, x = 800, y = 800, dt = dt_rad1)
    rs1, thetas1, phis1 = radar1.gen_data(position_data1)
    noisy_rs1, noisy_thetas1, noisy_phis1 = radar1.sense(rs1, thetas1, phis1)
    xs_from_rad1, ys_from_rad1, zs_from_rad1 = radar1.radar2cartesian(noisy_rs1, noisy_thetas1, noisy_phis1)

    radar1_values = np.array(list(zip(noisy_rs1, noisy_thetas1, noisy_phis1)))
    # print("Noisy radar1 values: \n{0}\n".format(radar1_values[:10]))
    radar1_computed_values = np.array(list(zip(xs_from_rad1, ys_from_rad1, zs_from_rad1)))
    # print("radar1 computed position values: \n{0}\n".format(radar1_computed_values[:10]))
    label_measurements1 = radar1.compute_measurements(position_data1)
    # print("radar2 computed measurements info: \n{0}\n".format(measurements_info2[:10]))

    # Radar 2: Standard radar
    radar2 = FrequencyRadar(tag = 1, x = 800, y = 200, dt = dt_rad2,
                            r_noise_std = 5., theta_noise_std = 0.005, phi_noise_std = 0.005)
    rs2, thetas2, phis2 = radar2.gen_data(position_data2)
    noisy_rs2, noisy_thetas2, noisy_phis2 = radar2.sense(rs2, thetas2, phis2)
    xs_from_rad2, ys_from_rad2, zs_from_rad2 = radar2.radar2cartesian(noisy_rs2, noisy_thetas2, noisy_phis2)

    radar2_values = np.array(list(zip(noisy_rs2, noisy_thetas2, noisy_phis2)))
    # print("Noisy radar2 values: \n{0}\n".format(radar2_values[:10]))
    radar2_computed_values = np.array(list(zip(xs_from_rad2, ys_from_rad2, zs_from_rad2)))
    # print("radar2 computed position values: \n{0}\n".format(radar2_computed_values[:10]))
    label_measurements2    = radar2.compute_measurements(position_data2)
    # print("radar2 computed measurements info: \n{0}\n".format(measurements_info2[:10]))

    # Combination of the measurements
    # radar_values          = np.concatenate((radar1_values,radar2_values),axis = 1)
    # radar_computed_values = np.concatenate((radar1_computed_values,radar2_computed_values),axis = 1)
    label_measurements    = sorted(label_measurements1 + label_measurements2)
    # ==========================================================================
    # ====================== Radar filter generation ===========================
    # Filter: constant velocity
    radars = [radar1,radar2]
    radar_filter_ta = MultipleFreqRadarsFilter(dim_x = 9, dim_z = 6, q = 400.,
                                               radars = radars, model = RadarFilterTA,
                                               x0 = 100, y0=100)
    est_xs_ta, est_ys_ta, est_zs_ta = [],[],[]
    for label_val in label_measurements:
        radar_filter_ta.predict()
        radar_filter_ta.update(label_val)
        est_xs_ta.append(radar_filter_ta.x[0,0])
        est_ys_ta.append(radar_filter_ta.x[3,0])
        est_zs_ta.append(radar_filter_ta.x[6,0])
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs1, ys1, zs1, label='Real position',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad1, ys_from_rad1, zs_from_rad1,color='b',marker='o',alpha = 0.3, label = 'Radar1 measurements')
    ax.scatter(xs_from_rad2, ys_from_rad2, zs_from_rad2,color='m',marker='o',alpha = 0.3, label = 'Radar2 measurements')
    ax.plot(est_xs_ta, est_ys_ta, est_zs_ta,color='orange', label = 'Estimation-TA')
    ax.scatter(radar1.x,radar1.y,radar1.z,color='r', label = 'Radar1')
    ax.scatter(radar2.x,radar2.y,radar2.z,color='g', label = 'Radar2')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
