# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 11:47:32 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from filterpy.kalman            import IMMEstimator
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import MultipleRadarsFilter, RadarFilterCV, RadarFilterCA, RadarFilterCT
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_takeoff()
    x0=states[0,0]
    y0=states[0,3]
    z0=states[0,6]

    # ==========================================================================
    # ======================== Radars data generation ===========================
    # Radar 1: Precision radar
    radar1 = Radar(x = 500, y = 500)

    # Radar 2: Standard radar
    radar2 = Radar(x = 0, y = 1000,
                   r_noise_std = 5., theta_noise_std = 0.005, phi_noise_std = 0.005)

    # ==========================================================================
    # ========================= IMM generation =================================
    radars = [radar1,radar2]
    radar_filter_cv = MultipleRadarsFilter(dim_x = 9, dim_z = 6, q = 1.,
                                           radars = radars, model = RadarFilterCV,
                                           x0 = x0, y0 = y0, z0 = z0)
    radar_filter_ca = MultipleRadarsFilter(dim_x = 9, dim_z = 6, q = 400.,
                                           radars = radars, model = RadarFilterCA,
                                           x0 = x0, y0 = y0, z0 = z0)
    radar_filter_ct = MultipleRadarsFilter(dim_x = 9, dim_z = 6, q = 350.,
                                           radars = radars, model = RadarFilterCT,
                                           x0 = x0, y0 = y0, z0 = z0)
    radar_filter_ta = MultipleRadarsFilter(dim_x = 9, dim_z = 6, q = 250.,
                                           radars = radars, model = RadarFilterTA,
                                           x0 = x0, y0 = y0, z0 = z0)
    filters = [radar_filter_cv, radar_filter_ca, radar_filter_ct, radar_filter_ta]
    mu = [0.25, 0.25, 0.25, 0.25]
    trans = np.array([[0.997, 0.001, 0.001, 0.001],
                      [0.050, 0.850, 0.050, 0.050],
                      [0.001, 0.001, 0.997, 0.001],
                      [0.001, 0.001, 0.001, 0.997]])
    imm = IMMEstimator(filters, mu, trans)

    benchmark_imm4 = Benchmark(radars = radars,radar_filter = imm,states = states)
    benchmark_imm4.launch_benchmark(with_nees = True)
