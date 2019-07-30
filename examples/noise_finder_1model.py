# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:05:34 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, Track
from fdia_simulation.helpers    import CSVWriter
from fdia_simulation.filters    import (RadarFilterCV, MultipleRadarsFilterCV,
                                        RadarFilterCA, MultipleRadarsFilterCA,
                                        RadarFilterCT, MultipleRadarsFilterCT,
                                        RadarFilterTA, MultipleRadarsFilterTA)
from fdia_simulation.benchmarks import NoiseFinder1Radar, NoiseFinder2Radars



writer = CSVWriter()
radar1 = Radar(x=2000,y=2000)
radar2 = Radar(x=1000,y=1000)
radars = [radar1, radar2]
trajectory = Track()
states = trajectory.gen_takeoff()
x0=states[0,0]
y0=states[0,3]
z0=states[0,6]

print('=================================================================')
print('===================== Constant Acceleration =====================')
print('========================== 1 Radar ==============================')

filter_ca = RadarFilterCA
noise_finder_ca = NoiseFinder1Radar(radar1, states, filter_ca)
noise_finder_ca.launch_benchmark()
best_value_ca = noise_finder_ca.best_value()
print('Best value for CA&1Radar:{0}'.format(best_value_ca))
writer.write_row('CA&1Radar',str(best_value_ca))

print('========================== 2 Radars =============================')

filter_mca = MultipleRadarsFilterCA
noise_finder_mca = NoiseFinder2Radars(radars, states, filter_mca)
noise_finder_mca.launch_benchmark()
best_value_mca = noise_finder_mca.best_value()
print('Best value for CA&2Radars:{0}'.format(best_value_mca))
writer.write_row('CA&2Radars',str(best_value_mca))

print('=================================================================')
print('====================== Constant Velocity ========================')
print('========================== 1 Radar ==============================')

filter_cv = RadarFilterCV
noise_finder_cv = NoiseFinder1Radar(radar1, states, filter_cv)
noise_finder_cv.launch_benchmark()
best_value_cv = noise_finder_cv.best_value()
print('Best value for CV&1Radar:{0}'.format(best_value_cv))
writer.write_row('CV&1Radar',str(best_value_cv))

print('========================== 2 Radars =============================')

filter_mcv = MultipleRadarsFilterCV
noise_finder_mcv = NoiseFinder2Radars(radars, states, filter_mcv)
noise_finder_mcv.launch_benchmark()
best_value_mcv = noise_finder_mcv.best_value()
print('Best value for CV&2Radars:{0}'.format(best_value_mcv))
writer.write_row('CV&2Radars',str(best_value_mcv))

print('=================================================================')
print('======================= Constant Turn  ==========================')
print('========================== 1 Radar ==============================')

filter_ct = RadarFilterCT
noise_finder_ct = NoiseFinder1Radar(radar1, states, filter_ct)
noise_finder_ct.launch_benchmark()
best_value_ct = noise_finder_ct.best_value()
print('Best value for CT&1Radar:{0}'.format(best_value_ct))
writer.write_row('CT&1Radar',str(best_value_ct))

print('========================== 2 Radars =============================')

filter_mct = MultipleRadarsFilterCT
noise_finder_mct = NoiseFinder2Radars(radars, states, filter_mct)
noise_finder_mct.launch_benchmark()
best_value_mct = noise_finder_mct.best_value()
print('Best value for CT&2Radars:{0}'.format(best_value_mct))
writer.write_row('CT&2Radars',str(best_value_mct))

print('=================================================================')
print('==================== Thrust Acceleration ========================')
print('========================== 1 Radar ==============================')

filter_ta = RadarFilterTA
noise_finder_ta = NoiseFinder1Radar(radar1, states, filter_ta)
noise_finder_ta.launch_benchmark()
best_value_ta = noise_finder_ta.best_value()
print('Best value for TA&1Radar:{0}'.format(best_value_ta))
writer.write_row('TA&1Radar',str(best_value_ta))

print('========================== 2 Radars =============================')

filter_mta = MultipleRadarsFilterTA
noise_finder_mta = NoiseFinder2Radars(radars, states, filter_mta)
noise_finder_mta.launch_benchmark()
best_value_mta = noise_finder_mta.best_value()
print('Best value for TA&2Radars:{0}'.format(best_value_mta))
writer.write_row('TA&2Radars',str(best_value_mta))
