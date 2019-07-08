# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:50:12 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.attackers.mo_attacker         import ExtendedMoAttacker
from fdia_simulation.filters.radar_filter_cv       import RadarFilterCV
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
    # ==========================================================================
    # ====================== Radar filter generation ===========================
    # Filter: constant velocity
    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 400., radar = radar)
    est_xs_cv, est_ys_cv, est_zs_cv = [],[],[]
    for val in radar_values:
        radar_filter_cv.predict()
        radar_filter_cv.update(val)
        est_xs_cv.append(radar_filter_cv.x[0,0])
        est_ys_cv.append(radar_filter_cv.x[3,0])
        est_zs_cv.append(radar_filter_cv.x[6,0])
    # ==========================================================================
    # ============================ Attacker generation =========================

    # mo_attacker = ExtendedMoAttacker(radar_filter_cv)
    # sample_nb = len(est_xs_cv)
    #
    # # Computation of the whole attack sequence
    # zas, Gamma = mo_attacker.compute_attack_sequence(attack_size = sample_nb, pos_value = 1, logs = True)

    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='Real position',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad, ys_from_rad, zs_from_rad,color='b',marker='o',alpha = 0.3, label = 'Radar measurements')
    ax.plot(est_xs_cv, est_ys_cv, est_zs_cv,color='orange', label = 'Estimation-CV')
    ax.scatter(radar.x,radar.y,radar.z,color='r', label = 'Radar')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
