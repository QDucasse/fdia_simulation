# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:36:32 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models.moving_target          import Command
from fdia_simulation.models.maneuvered_aircraft    import ManeuveredAircraft
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.attackers.mo_attacker         import MoAttacker
from filterpy.kalman                               import KalmanFilter, ExtendedKalmanFilter


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    # Route generation example with a ManeuveredAircraft
    sensor_std = 1.
    headx_cmd = Command('headx',0,0,0)
    headz_cmd = Command('headz',0,0,0)
    vel_cmd   = Command('vel',1,0,0)
    aircraft  = ManeuveredAircraft(x0 = 1, y0 = 1, z0=1, v0 = 0, hx0 = 0, hz0 = 0, command_list = [headx_cmd, headz_cmd, vel_cmd])
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
    aircraft.change_command("headx",45, 25)
    aircraft.change_command("headz",90, 25)

    # Second phase -> Take off
    for i in range(30):
        x, y, z = aircraft.update()
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Change in commands -> Steady state
    aircraft.change_command("headx",0, 25)
    aircraft.change_command("headz",180, 25)

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
    radar1 = Radar(x=800,y=800)
    rs, thetas, phis = radar1.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar1.sense(rs, thetas, phis)
    xs_from_rad1, ys_from_rad1, zs_from_rad1 = radar1.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)

    radar1_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    radar1_computed_values = np.array(list(zip(xs_from_rad1, ys_from_rad1, zs_from_rad1)))

    # Radar 2
    radar2 = Radar(x=-1000,y=500)
    rs, thetas, phis = radar2.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar2.sense(rs, thetas, phis)
    xs_from_rad2, ys_from_rad2, zs_from_rad2 = radar2.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)

    radar2_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    radar2_computed_values = np.array(list(zip(xs_from_rad2, ys_from_rad2, zs_from_rad2)))
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure()
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='plot test',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad1, ys_from_rad1, zs_from_rad1,color='b',marker='o',alpha = 0.3, label = 'Radar 1 measurements')
    ax.scatter(radar1.x,radar1.y,radar1.z,color='r', label = 'Radar 1')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.scatter(xs_from_rad2, ys_from_rad2, zs_from_rad2,color='g',marker='o',alpha = 0.3, label = 'Radar 2 measurements')
    ax.scatter(radar2.x,radar2.y,radar2.z,color='magenta',label='Radar 2')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
    # ==========================================================================
    # ======================== Filter generation  ==============================








    # ==========================================================================
    # ======================== Attacker generation  ============================










    # ==========================================================================
    # ======================== State estimation ================================






    # ==========================================================================
    # ======================= Fault detection ==================================





    # ==========================================================================
    # ========================== Comparison ====================================
