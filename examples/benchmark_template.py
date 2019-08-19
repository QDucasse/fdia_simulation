# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:09:24 2019

@author: qde
"""

from fdia_simulation.models            import Radar, PeriodRadar, Track
from fdia_simulation.filters           import (RadarFilterCA, MultipleRadarsFilterCA, MultiplePeriodRadarsFilterCA,
                                               RadarFilterCV, MultipleRadarsFilterCV, MultiplePeriodRadarsFilterCV,
                                               RadarFilterCT, MultipleRadarsFilterCT, MultiplePeriodRadarsFilterCT,
                                               RadarFilterTA, MultipleRadarsFilterTA, MultiplePeriodRadarsFilterTA)
from fdia_simulation.attackers         import (Attacker,DOSAttacker,DriftAttacker,
                                               PeriodAttacker,DOSPeriodAttacker,DriftPeriodAttacker)
from fdia_simulation.benchmarks        import Benchmark
from fdia_simulation.anomaly_detectors import ChiSquareDetector, EuclidianDetector

if __name__ == "__main__":

    # ==========================================================================
    # ================== Position generation for the aircraft ==================
    # ==========================================================================
    trajectory = Track()
    states = trajectory.gen_landing()
    x0,y0,z0 = trajectory.initial_position(states)

    # ==========================================================================
    # ========================= Radar(s) Generation ============================
    # ==========================================================================
    ## =================== Standard Radars (same data rates)
    ## Specify your own radar positions and beliefs over measurements!
    # Radar 1: Precision radar
    radar1 = Radar(x = -6000, y = 10000)

    # Radar 2: Standard radar
    radar2 = Radar(x = 1000, y = 8000,
                   r_std = 5., theta_std = 0.005, phi_std = 0.005)

    radars = [radar1,radar2]

    ## ================== Period Radars (different data rates)
    # Radar 1: Precision radar
    dt_rad1 = 0.1
    pradar1 = PeriodRadar(x = -6000, y = 10000, dt = dt_rad1)

    # Radar 2: Standard radar
    pradar2 = PeriodRadar(x = 1000, y = 8000, dt = dt_rad2,
                            r_std = 5., theta_std = 0.005, phi_std = 0.005)
    dt_rad2 = 0.4

    pradars = [pradar1,pradar2]
    # ==========================================================================
    # ========================= Detector Generation ============================
    # ==========================================================================
    ## Comment the unused detectors!
    detector = None
    detector = ChiSquareDetector()
    detector = EuclidianDetector()

    # ==========================================================================
    # ========================= Filter(s) Generation ===========================
    # ==========================================================================
    ## Specify the model and process noise you want!
    ## Comment the unused filters!

    ## One model filters
    radar_filter = RadarFilterCV(q = 2300., radar = radar1, detector = detector,
                                 x0 = x0, y0 = y0, z0 = z0)

    radar_filter = MultipleRadarsFilterCV(q = 3040., radars = radars, detector = detector,
                                          x0 = x0, y0 = y0, z0 = z0)

    radar_filter = MultiplePeriodRadarsFilterCV(q = 3920., radars = pradars, detector = detector,
                                                 x0 = x0, y0 = y0, z0 = z0)

    # ==========================================================================
    # ========================= Attacker Generation ============================
    # ==========================================================================
    ## Comment the unused attackers!
    ## Specify your own t0, time, radar_pos and radar!
    ## note radar1 => radar_pos = 0, radar2 => radar_pos = 1, etc.

    ## No attacker
    attacker = None

    ## Attacker for 1 or 2 Radars
    attacker = DOSAttacker(filter = radar_filter,
                                  t0 = 200, time = 100, radar_pos = 1)

    attacker = DriftAttacker(filter = radar_filter, t0 = 200,
                             time = 100, radar_pos = 1, radar = radar2)

    ## Attackers for 2 Period Radars
    attacker = DriftPeriodAttacker(filter = radar_filter_cv, t0 = 200, time = 2000,
                                   radar = pradar2, radar_pos = 1)

    attacker = DOSPeriodAttacker(filter = radar_filter, t0 = 200, time = 2000,
                                        radar = pradar2, radar_pos = 1)

    # ==========================================================================
    # ========================= Benchmark Wrapping =============================
    # ==========================================================================
    ## Comment the unused benchmark!
    benchmark_standard = Benchmark(radars = radars,radar_filter = radar_filter,
                                   states = states, attacker = attacker)
    benchmark_ca.launch_benchmark(with_nees = True)

    benchmark_period = Benchmark(radars = pradars,radar_filter = radar_filter,
                                 states = states, attacker = attacker)
    benchmark_ca.launch_benchmark(with_nees = True)
