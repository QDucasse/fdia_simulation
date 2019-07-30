# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:05:34 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, Track
from fdia_simulation.helpers    import CSVWriter
from fdia_simulation.filters    import (RadarFilterCV, MultipleRadarsFilterCV, MultipleFreqRadarsFilterCV,
                                        RadarFilterCA, MultipleRadarsFilterCA, MultipleFreqRadarsFilterCA,
                                        RadarFilterCT, MultipleRadarsFilterCT, MultipleFreqRadarsFilterCT,
                                        RadarFilterTA, MultipleRadarsFilterTA, MultipleFreqRadarsFilterTA)
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
FILTERS_2_FRADARS = [MultipleFreqRadarsFilterCV,
                     MultipleFreqRadarsFilterCA,
                     MultipleFreqRadarsFilterCT,
                     MultipleFreqRadarsFilterTA]

# Initialization of our components:
## Writer
writer = CSVWriter()
## Radars
radar1 = Radar(x=2000,y=2000)
radar2 = Radar(x=1000,y=1000)
radars = [radar1, radar2]
##States
trajectory = Track()
states = trajectory.gen_takeoff()
x0=states[0,0]
y0=states[0,3]
z0=states[0,6]

# Loop over every list and output + write results
for filter in FILTERS_1_RADAR:
    name = filter.__name__[-2:]
    print('=================================================================')
    print('=========================== '+ name +'-1 Radar ==========================')
    noise_finder = NoiseFinder1Radar(radar1, states, filter)
    noise_finder.launch_benchmark()
    best_value   = noise_finder.best_value()
    print(('Best value for '+ name +'-1Radar:{0}').format(best_value))
    writer.write_row(name+'-1Radar',str(best_value))

for filter in FILTERS_2_RADARS:
    name = filter.__name__[-2:]
    print('=================================================================')
    print('=========================== '+ name +'-2 Radars =========================')
    noise_finder = NoiseFinder2Radars(radars, states, filter)
    noise_finder.launch_benchmark()
    best_value   = noise_finder.best_value()
    print(('Best value for '+ name +'-2Radars:{0}').format(best_value))
    writer.write_row(name+'-2Radars',str(best_value))

for filter in FILTERS_2_FRADARS:
    name = filter.__name__[-2:]
    print('=================================================================')
    print('========================== '+ name +'-2 FRadars =========================')
    noise_finder = NoiseFinder2Radars(radars, states, filter)
    noise_finder.launch_benchmark()
    best_value   = noise_finder.best_value()
    print(('Best value for '+ name +'-2FRadars:{0}').format(best_value))
    writer.write_row(name+'-2FRadars',str(best_value))
