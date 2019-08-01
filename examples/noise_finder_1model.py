# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:05:34 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, PeriodRadar, Track
from fdia_simulation.helpers    import CSVWriter
from fdia_simulation.filters    import (RadarFilterCV, MultipleRadarsFilterCV, MultiplePeriodRadarsFilterCV,
                                        RadarFilterCA, MultipleRadarsFilterCA, MultiplePeriodRadarsFilterCA,
                                        RadarFilterCT, MultipleRadarsFilterCT, MultiplePeriodRadarsFilterCT,
                                        RadarFilterTA, MultipleRadarsFilterTA, MultiplePeriodRadarsFilterTA)
from fdia_simulation.benchmarks import NoiseFinder1Radar, NoiseFinder2Radars

# Filters to be tested for 1 radar
FILTERS_1_RADAR   = [RadarFilterCV,
                     RadarFilterCA,
                     RadarFilterCT,
                     RadarFilterTA]

# Filters to be tested for 2 radars
FILTERS_2_RADARS  = [MultipleRadarsFilterCV,
                     MultipleRadarsFilterCA,
                     MultipleRadarsFilterCT,
                     MultipleRadarsFilterTA]

# Filters to be tested for 2 radars with different data rates
FILTERS_2_FRADARS = [MultiplePeriodRadarsFilterCV,
                     MultiplePeriodRadarsFilterCA,
                     MultiplePeriodRadarsFilterCT,
                     MultiplePeriodRadarsFilterTA]

# Initialization of our components:
## Writer
writer = CSVWriter()
## Radars
## For 1&2 radars same data rate
radar1 = Radar(x = -6000, y = 10000)
radar2 = Radar(x = 1000, y = 8000)
radars = [radar1, radar2]
## Different data rates radars
dt1 = 0.1
dt2 = 0.4
fradar1 = PeriodRadar(x = -6000,y = 10000,dt = dt1)
fradar2 = PeriodRadar(x = 1000, y = 8000, dt = dt2,
                         r_std = 5., theta_std = 0.005, phi_std = 0.005)
fradars = [fradar1, fradar2]

##States
trajectory = Track()
states = trajectory.gen_landing()
x0=states[0,0]
y0=states[0,3]
z0=states[0,6]

# Loop over every list and output + write results
for filter in FILTERS_1_RADAR:
    name = filter.__name__[-2:]
    print('=================================================================')
    print('=========================== '+ name +'-1 Radar ==========================')
    noise_finder = NoiseFinder1Radar(radar1, states, filter, nb_iterations = 3)
    noise_finder.launch_benchmark()
    best_value   = noise_finder.best_value()
    print(('Best value for '+ name +'-1Radar:{0}').format(best_value))
    writer.write_row(name+'-1Radar',str(best_value))

for filter in FILTERS_2_RADARS:
    name = filter.__name__[-2:]
    print('=================================================================')
    print('=========================== '+ name +'-2 Radars =========================')
    noise_finder = NoiseFinder2Radars(radars, states, filter, nb_iterations = 3)
    noise_finder.launch_benchmark()
    best_value   = noise_finder.best_value()
    print(('Best value for '+ name +'-2Radars:{0}').format(best_value))
    writer.write_row(name+'-2Radars',str(best_value))

for filter in FILTERS_2_FRADARS:
    name = filter.__name__[-2:]
    print('=================================================================')
    print('========================== '+ name +'-2 FRadars =========================')
    noise_finder = NoiseFinder2Radars(fradars, states, filter, nb_iterations = 3)
    noise_finder.launch_benchmark()
    best_value   = noise_finder.best_value()
    print(('Best value for '+ name +'-2FRadars:{0}').format(best_value))
    writer.write_row(name+'-2FRadars',str(best_value))
