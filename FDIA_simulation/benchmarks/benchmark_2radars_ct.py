# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 11:47:42 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models.radar            import Radar
from fdia_simulation.models.tracks           import Track
from fdia_simulation.attackers.mo_attacker   import MoAttacker
from fdia_simulation.filters.radar_filter_ct import RadarFilterCT
from fdia_simulation.filters.mradar_filter   import MultipleRadarsFilter


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    xs, ys, zs = trajectory.gen_takeoff()
    position_data = np.array(list(zip(xs,ys,zs)))
    # ==========================================================================
    # ======================== Radars data generation ===========================
    # Radar 1
    radar1 = Radar(x=800,y=800)
    rs1, thetas1, phis1 = radar1.gen_data(position_data)
    noisy_rs1, noisy_thetas1, noisy_phis1 = radar1.sense(rs1, thetas1, phis1)
    xs_from_rad1, ys_from_rad1, zs_from_rad1 = radar1.radar2cartesian(noisy_rs1, noisy_thetas1, noisy_phis1)

    radar1_values = np.array(list(zip(noisy_rs1, noisy_thetas1, noisy_phis1)))
    # print("Noisy radar1 values: \n{0}\n".format(radar1_values[:10]))
    radar1_computed_values = np.array(list(zip(xs_from_rad1, ys_from_rad1, zs_from_rad1)))
    # print("radar1 computed position values: \n{0}\n".format(radar1_computed_values[:10]))

    # Radar 2
    radar2 = Radar(x=800,y=200)
    rs2, thetas2, phis2 = radar2.gen_data(position_data)
    noisy_rs2, noisy_thetas2, noisy_phis2 = radar2.sense(rs2, thetas2, phis2)
    xs_from_rad2, ys_from_rad2, zs_from_rad2 = radar2.radar2cartesian(noisy_rs2, noisy_thetas2, noisy_phis2)

    radar2_values = np.array(list(zip(noisy_rs2, noisy_thetas2, noisy_phis2)))
    # print("Noisy radar2 values: \n{0}\n".format(radar2_values[:10]))
    radar2_computed_values = np.array(list(zip(xs_from_rad2, ys_from_rad2, zs_from_rad2)))
    # print("radar2 computed position values: \n{0}\n".format(radar2_computed_values[:10]))
    radar_values          = np.concatenate((radar1_values,radar2_values),axis = 1)
    radar_computed_values = np.concatenate((radar1_computed_values,radar2_computed_values),axis = 1)
    # ==========================================================================
    # ====================== Radar filter generation ===========================
    # Filter: constant velocity
    radars = [radar1,radar2]
    radar_filter_ct = MultipleRadarsFilter(dim_x = 9, dim_z = 6, q = 400.,
                                           radars = radars, model = RadarFilterCT   ,
                                           x0 = 100, y0=100)
    est_xs_ct, est_ys_ct, est_zs_ct = [],[],[]
    for val in radar_values:
        radar_filter_ct.predict()
        radar_filter_ct.update(val)
        est_xs_ct.append(radar_filter_ct.x[0,0])
        est_ys_ct.append(radar_filter_ct.x[3,0])
        est_zs_ct.append(radar_filter_ct.x[6,0])
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='Real position',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad1, ys_from_rad1, zs_from_rad1,color='b',marker='o',alpha = 0.3, label = 'Radar1 measurements')
    ax.scatter(xs_from_rad2, ys_from_rad2, zs_from_rad2,color='m',marker='o',alpha = 0.3, label = 'Radar2 measurements')
    ax.plot(est_xs_ct, est_ys_ct, est_zs_ct,color='orange', label = 'Estimation-CT')
    ax.scatter(radar1.x,radar1.y,radar1.z,color='r', label = 'Radar1')
    ax.scatter(radar2.x,radar2.y,radar2.z,color='g', label = 'Radar2')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
