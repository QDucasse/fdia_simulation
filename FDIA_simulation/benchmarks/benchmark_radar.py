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
from filterpy.kalman                               import KalmanFilter


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
    print("Aircraft position:\n{0}\n".format(position_data[-25:,:]))
    # ==========================================================================
    # ======================== Radar data generation ===========================
    radar = Radar()
    rs, thetas, phis = radar.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
    xs_from_rad, ys_from_rad, zs_from_rad = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)

    radar_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    print("Noisy radar data:\n{0}\n".format(radar_values[-25:,:]))

    radar_computed_values = np.array(list(zip(xs_from_rad, ys_from_rad, zs_from_rad)))
    print("Estimated positions:\n{0}\n".format(radar_computed_values[-25:,:]))
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure()
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='plot test',color='k',linestyle='dashed')
    ax.scatter(xs_from_rad, ys_from_rad, zs_from_rad,color='b',marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    # ==========================================================================
    # ======================== Filter generation  ==============================








    # ==========================================================================
    # ======================== Attacker generation  ============================










    # ==========================================================================
