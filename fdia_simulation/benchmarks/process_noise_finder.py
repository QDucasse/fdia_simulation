# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:57:08 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import MultipleRadarsFilterCA
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark

class NoiseFinder1Radar(object):
    '''
    Implements a helper to determine the best process noise q to your model and
    configuration.
    Parameters
    ----------
    radar: Radar object
        The radar observing the system.

    states: numpy float array
        States of the system.

    filter: RadarFilterModel object
        Filter with a given model to be tested.

    nb_iterations: int
        Number of times the same simulation will be processed. Put other than 1
        if you think the randomness can generate unlucky faulty simulations.

    Attributes
    ----------
    Same as parameters +

    mean_nees: dictionary(key:float, value:float)
        Dictionary of every tested q (key) and its associated average nees (value).
    '''

    # List of tested values:
    # 0.01, 0.02 ... 0.09
    # 0.1 ,  0.2 ... 0.9
    # 1   ,    2 ... 9
    # 10  ,   20 ... 4000

    TO_TEST = list(np.linspace(0.01,0.09,num=9)) + \
              list(np.linspace(0.1,0.9,num=9))   + \
              list(np.linspace(1,9,num=9))       + \
              list(np.linspace(10,4000,num=400))

    def __init__(self,radar,states,filter,nb_iterations = 1):
        self.radar         = radar
        self.states        = states
        self.filter        = filter
        self.nb_iterations = nb_iterations
        self.means_nees    = {}

    def compute_nees(self,q):
        '''
        Computes the average Normalized Estimated Error Squared (nees) for a given
        filter and process noise q.
        Parameters
        ----------
        q: float
            Process noise to be tested.

        Returns
        -------
        mean_nees: float
            Average nees of the tested filter/process noise.

        '''
        x0 = self.states[0,0]
        y0 = self.states[0,3]
        z0 = self.states[0,6]
        filter = self.filter(dim_x = 9, dim_z = 3, radar = self.radar, q = q, x0 = x0, y0 = y0, z0 = z0)
        benchmark = Benchmark(radars = self.radar, radar_filter = filter, states = self.states)
        benchmark.launch_benchmark(with_nees = True, plot = False)
        return benchmark.nees

    def iterate_same_simulation(self,q):
        '''
        Iterates the benchmark a given number of times to avoid unlucky singular
        behavior.
        Parameters
        ----------
        q: float
            Tested process noise.

        Returns:
        --------
        means_nees: float iterable
            Means of the nees of the iterations.
        '''
        one_q_means_nees = []
        for _ in range(self.nb_iterations): # in case of unlucky simulations
            mean_nees = np.mean(self.compute_nees(q))
            one_q_means_nees.append(mean_nees)
        return one_q_means_nees

    def launch_benchmark(self):
        '''
        Tests the filter over all the qs in TO_TEST. Adds to the keys q the
        average nees as value.
         e.g.: self.mean_nees = {q: mean_nees for q}
        Parameters
        ----------
        filter: RadarFilterModel object
            Filter model to be tested.
        '''
        count = 0
        for q in self.TO_TEST:  # Loop over all the values that should be tested
            current_mean_nees = self.iterate_same_simulation(q)
            self.means_nees[q] = min(current_mean_nees)

            # Proof of life
            if (count%10 == 0): print("Ongoing: step n°{0}/430".format(count))
            count += 1
        print("Ongoing: step n°430/430"

    def best_value(self):
        '''
        Returns the q with the minimal nees processed by the benchmark.
        Returns
        -------
        best_value: float
            q value with the smallest associated average nees.
        '''
        best_value = min(self.means_nees, key=self.means_nees.get)
        return best_value

class NoiseFinder2Radars(NoiseFinder1Radar):
    '''
    Implements a helper to find the correct noise in a 2 radars case.

    Notes
    -----
    Please see NoiseFinder1Radar help for more information.
    '''
    def __init__(self,radars,states,filter,nb_iterations = 1):
        self.radars        = radars
        self.states        = states
        self.filter        = filter
        self.nb_iterations = nb_iterations
        self.means_nees    = {}

    def compute_nees(self,q):
        '''
        Computes the average Normalized Estimated Error Squared (nees) for a given
        filter and process noise q.
        Parameters
        ----------
        q: float
            Process noise to be tested.

        Returns
        -------
        mean_nees: float
            Average nees of the tested filter/process noise.

        '''
        x0 = self.states[0,0]
        y0 = self.states[0,3]
        z0 = self.states[0,6]
        filter = self.filter(dim_x = 9, dim_z = 3, radars = self.radars, q = q, x0 = x0, y0 = y0, z0 = z0)
        benchmark = Benchmark(radars = self.radars, radar_filter = filter, states = self.states)
        benchmark.launch_benchmark(with_nees = True, plot = False)
        return benchmark.nees
