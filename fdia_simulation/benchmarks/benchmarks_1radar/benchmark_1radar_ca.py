# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:50:21 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import RadarFilterCA
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark


if __name__ == "__main__":
    trajectory = Track()
    states = trajectory.gen_takeoff()
    x0=states[0,0]
    y0=states[0,3]
    z0=states[0,6]
    print('states: \n{0}\n'.format(states))
    print(np.shape(states))

    radar = Radar(x=2000,y=2000)
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 3600., x0=x0,y0=y0,z0=z0, radar = radar)

    benchmark_ca = Benchmark(radars = radar, radar_filter = radar_filter_ca,states = states)
    benchmark_ca.launch_benchmark(with_nees = True)



    # # ==========================================================================
    # # ======================== Radar data generation ===========================
    # trajectory = Track()
    # states = trajectory.gen_takeoff()
    # # print(states)
    # xs = states[:,0]
    # ys = states[:,3]
    # zs = states[:,6]
    # position_data = np.array(list(zip(xs,ys,zs)))
    # position_data = position_data[::10]
    # # print('position data: \n{0}\n'.format(position_data))
    #
    # radar = Radar(x=0,y=0)
    # rs, thetas, phis = radar.gen_data(position_data)
    # noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
    # xs_from_rad, ys_from_rad, zs_from_rad = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)
    #
    # radar_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    # radar_computed_values = np.array(list(zip(xs_from_rad, ys_from_rad, zs_from_rad)))
    # # print('computed positions: \n{0}\n'.format(radar_computed_values))
    # # print("Noisy radar values: \n{0}\n".format(radar_values[:10]))
    # # print("Radar computed position values: \n{0}\n".format(radar_computed_values[:10]))
    # # # ==========================================================================
    # # # ====================== Radar filter generation ===========================
    # # Filter: constant acceleration
    #
    # radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)
    # # print('F: \n{0}\n'.format(radar_filter_ca.F))
    # # print('Q: \n{0}\n'.format(radar_filter_ca.Q))
    # # print('x: \n{0}\n'.format(radar_filter_ca.x))
    #
    # est_xs_ca, est_ys_ca, est_zs_ca = [],[],[]
    # for val in radar_values:
    #     # print('Current_value: \n{0}\n'.format(val))
    #     # print('Computed position: \n{0}\n'.format(radar.radar2cartesian([val[0]],[val[1]],[val[2]])))
    #     radar_filter_ca.predict()
    #     radar_filter_ca.update(val)
    #     # print('New state: \n{0}\n'.format(radar_filter_ca.x))
    #     est_xs_ca.append(radar_filter_ca.x[0,0])
    #     est_ys_ca.append(radar_filter_ca.x[3,0])
    #     est_zs_ca.append(radar_filter_ca.x[6,0])
    # # print('estimated xs: \n{0}\n'.format(est_xs_ca))
    # # print('estimated ys: \n{0}\n'.format(est_ys_ca))
    # # print('estimated zs: \n{0}\n'.format(est_zs_ca))
    # # ==========================================================================
    # # =============================== Plotting =================================
    # fig = plt.figure(1)
    # plt.rc('font', family='serif')
    # ax = fig.gca(projection='3d')
    # ax.plot(xs, ys, zs, label='Real position',color='k',linestyle='dashed')
    # ax.scatter(xs_from_rad, ys_from_rad, zs_from_rad,color='b',marker='o',alpha = 0.3, label = 'Radar measurements')
    # ax.plot(est_xs_ca, est_ys_ca, est_zs_ca,color='m', label = 'Estimation-CA')
    # ax.scatter(radar.x,radar.y,radar.z,color='r', label = 'Radar')
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # ax.legend()
    # plt.show()
