# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:06:12 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from copy import deepcopy
from filterpy.kalman import IMMEstimator
from fdia_simulation.models.moving_target          import Command
from fdia_simulation.models.maneuvered_aircraft    import ManeuveredAircraft
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.attackers.mo_attacker         import MoAttacker
from fdia_simulation.filters.radar_filter_ca       import RadarFilterCA
from fdia_simulation.models.tracks                 import Track


class Benchmark1Radar(object):
    r'''Implements a benchmark to create an estimation of a trajectory detected
    from a set of radars and estimated by a set of filters.
    Parameters
    ----------
    radar: Radar iterable
        List of radars observing the moving target.

    radar_filter: RadarModel instance
        Filter or IMM that will estimate the state of the observed system.

    states numpy array iterable
        List of the true states of the observed system.
    '''
    def __init__(self,radar,radar_filter,states):
        self.radar        = radar
        self.radar_filter = radar_filter
        self.states       = states
        xs,ys,zs          = states[:,0], states[:,3], states[:,6]
        self.pos_data     = np.array(list(zip(xs,ys,zs)))
        # Check if the filter is an IMM or not
        self.filter_is_imm = False
        if type(self.radar_filter) == IMMEstimator:
            self.filter_is_imm = True
        # Actual values to be plotted
        self.measured_positions  = []
        self.estimated_positions = []
        self.nees                = []
        self.probs               = []
        # IMM model names
        self.radar_filters_names = []

    def gen_data_set(self):
        '''
        Generates the measured data from the radars.
        Returns
        -------
        measured_values: numpy array.
            Array of the different measurements made by the radar [r,theta,phi].

        Notes
        -----
        The function adds measurement noise to the measurements from the radars.
        It also fills the computed positions to allow plotting of the sensed
        positions.
        '''
        rs, thetas, phis = self.radar.gen_data(self.pos_data)
        noisy_rs, noisy_thetas, noisy_phis = self.radar.sense(rs, thetas, phis)
        xs, ys, zs = self.radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)
        self.measured_values    = np.array(list(zip(noisy_rs,noisy_thetas,noisy_phis)))
        self.measured_positions = np.array(list(zip(xs,ys,zs)))
        return self.measured_values

    def process_filter(self,measurements = None,with_nees = False):
        '''
        Launches the filter cycles of predict/update.
        Parameters
        ----------
        with_nees: boolean
            Triggers the computation of NEES (Normalized Estimated Error Squared).

        Returns
        -------
        est_states: numpy array
            Estimated states of the observed system.
        '''
        if measurements is None:
            measurements = self.measured_values

        # Initialization of the lists of:
        # estimated states: results of the estimator on state space vector
        # nees: Normalized Estimated Error Squared (if mode triggered)
        est_states, nees, probs = [],[],[]
        # Scrolling through the measurements
        for i,measurement in enumerate(self.measured_values):
            self.radar_filter.predict()
            self.radar_filter.update(measurement)
            current_state = deepcopy(self.radar_filter.x)
            est_states.append(current_state)
            # print('Estimate states vector:\n{0}\n'.format(est_states))
            if with_nees:
                # Computation of the error between true and estimated states
                states_tilde = np.subtract(est_states[i],np.reshape(self.states[i,:],(-8,1)))
                nees.append(*(states_tilde.T@inv(self.radar_filter.P)@states_tilde))

            if self.filter_is_imm:
                probs.append(self.radar_filter.mu)
        # Conversion in arrays

        est_states = np.array(est_states)
        nees       = np.array(nees)
        probs      = np.array(probs)
        # Extraction of the position (for plotting)
        est_xs     = est_states[:,0]
        est_ys     = est_states[:,3]
        est_zs     = est_states[:,6]
        self.estimated_positions = np.concatenate((est_xs,est_ys,est_zs),axis=1)
        self.nees  = nees
        self.probs = probs

    def generate_plotting_labels(self):
        '''
        Extracts the names of radar filter(s) for plotting purposes.
        '''
        # If the filter is an IMM, we need the name of each of the used models.
        if self.filter_is_imm:
            filter_models = self.radar_filter.filters
            for model in filter_models:
                model_name = 'Estimation-' + type(model).__name__[-2:]
                self.radar_filters_names.append(model_name)

        # Else, we have a simple model and therefore need only its name.
        else:
            model_name = 'Estimation-' + type(self.radar_filter).__name__[-2:]
            self.radar_filters_names.append(model_name)



    def plot(self):
        '''
        Plots all the computed data sets:
        - Real system positions
        - Measured system positions for each radar
        - Actual radar positions
        - Final estimated positions
        if the filter is an IMM:
        - Model probabilities
        '''
        real_xs = self.pos_data[:,0]
        real_ys = self.pos_data[:,1]
        real_zs = self.pos_data[:,2]

        measured_xs = self.measured_positions[:,0]
        measured_ys = self.measured_positions[:,1]
        measured_zs = self.measured_positions[:,2]

        estimated_xs = self.estimated_positions[:,0]
        estimated_ys = self.estimated_positions[:,1]
        estimated_zs = self.estimated_positions[:,2]

        fig = plt.figure(1)
        plt.rc('font', family='serif')
        ax = fig.gca(projection='3d')
        ax.plot(real_xs, real_ys, real_zs, label='Real trajectory',color='k',linestyle='dashed')
        # Radar measurements
        ax.scatter(measured_xs, measured_ys, measured_zs,marker='o',alpha = 0.3, label = 'Radar measurements')

        # Radar positions
        ax.scatter(self.radar.x,self.radar.y,self.radar.z, marker = 'x', label = 'Radar position')

        # Estimated positions
        ax.plot(estimated_xs,estimated_ys,estimated_zs,color='orange', label= 'Estimated trajectory')

        # Axis labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend()
        fig.show()

        if len(self.nees)>0:
            fig2 = plt.figure(2)
            plt.title('Normalized Estimation Error Squared (NEES)')
            plt.plot(self.nees)
            fig2.show()

        if self.filter_is_imm:
            fig3 = plt.figure(3)
            for i in range(len(self.probs[0,:])):
                plt.plot(self.probs[:,i],label = self.radar_filters_names[i])
            plt.xlabel('Time')
            plt.ylabel('Model probability')
            plt.title('Probabilities of the different models')
            plt.legend()
            fig2.show()


        plt.show()

    def launch_benchmark(self, with_nees = False):
        '''
        Launches the usual benchmark procedure:
        - Generates radar's data set
        - Porcesses the filter over the generated measurements
        - Plots the different graphs
        Parameters
        ----------
        with_nees: boolean
            Triggers the display of the evolution of the NEES.
        '''
        self.gen_data_set()
        self.process_filter(with_nees = with_nees)
        self.generate_plotting_labels()
        self.plot()
