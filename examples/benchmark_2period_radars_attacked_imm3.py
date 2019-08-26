# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 11:47:28 2019

@author: qde
"""
import numpy as np
from fdia_simulation.models            import PeriodRadar, Track
from fdia_simulation.filters           import (MultiplePeriodRadarsFilterCA,
                                               MultiplePeriodRadarsFilterCV,
                                               MultiplePeriodRadarsFilterCT,
                                               MultiplePeriodRadarsFilterTA,
                                               RadarIMM)
from fdia_simulation.attackers         import PeriodAttacker,DOSPeriodAttacker,DriftPeriodAttacker,CumulativeDriftPeriodAttacker
from fdia_simulation.benchmarks        import Benchmark
from fdia_simulation.anomaly_detectors import MahalanobisDetector,EuclidianDetector


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_landing()
    x0,y0,z0 = trajectory.initial_position(states)
    # ================================ Radar(s) ================================
    ## Period radars
    dt_rad1 = 0.2
    dt_rad2 = 0.05
    # Radar 1: Precision radar
    pradar1 = PeriodRadar(x = -6000, y = 10000, dt = dt_rad1)

    # Radar 2: Standard radar
    pradar2 = PeriodRadar(x = 1000, y = 8000, dt = dt_rad2,
                            r_std = 5., theta_std = 0.005, phi_std = 0.005)
    pradars = [pradar1,pradar2]

    # ================================ Detectors ===============================
    # detector = None
    # detector = MahalanobisDetector(error_rate = 0.05)
    detector = EuclidianDetector(error_rate = 0.05)

    # ================================ Filters =================================
    radar_filter_cv = MultiplePeriodRadarsFilterCV(q = 200.,detector = detector,
                                                   radars = pradars,
                                                   x0 = x0, y0 = y0, z0 = z0)
    radar_filter_ca = MultiplePeriodRadarsFilterCA(q = 100., detector = detector,
                                                   radars = pradars,
                                                   x0 = x0, y0 = y0, z0 = z0)
    radar_filter_ct = MultiplePeriodRadarsFilterCT(q = 350., detector = detector,
                                                   radars = pradars,
                                                   x0 = x0, y0 = y0, z0 = z0)
    filters = [radar_filter_cv, radar_filter_ca, radar_filter_ct]
    mu = [0.33, 0.33, 0.33]
    trans = np.array([[0.998, 0.001, 0.001],
                      [0.050, 0.900, 0.050],
                      [0.001, 0.001, 0.998]])
    imm = RadarIMM(filters, mu, trans)

    # =============================== Attackers ================================
    ## No attacker
    # attacker = None

    ## Attackers for 2 Period Radars
    # attacker = DriftPeriodAttacker(filter = imm, t0 = 500, time = 300,
    #                                attack_drift = np.array([[100,100,100]]).T,
    #                                radar = pradar2, radar_pos = 1)

    attacker = CumulativeDriftPeriodAttacker(filter = imm, t0 = 300, time = 2000,
                                   delta_drift = np.array([[0,0,1]]).T,
                                   radar = pradar2, radar_pos = 1)


    # =============================== Benchmark ================================
    benchmark = Benchmark(radars = pradars,radar_filter = imm,
                          states = states, attacker = attacker)
    benchmark.launch_benchmark(with_nees = True)
