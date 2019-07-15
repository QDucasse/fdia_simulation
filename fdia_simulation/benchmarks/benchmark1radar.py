# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:06:12 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from fdia_simulation.models.moving_target          import Command
from fdia_simulation.models.maneuvered_aircraft    import ManeuveredAircraft
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.attackers.mo_attacker         import MoAttacker
from fdia_simulation.filters.radar_filter_ca       import RadarFilterCA
from fdia_simulation.models.tracks                 import Track


class Benchmark1Radar(object):
    def __init__(self,radar,radar_filter,pos_data,states):
        self.radar        = radar
        self.radar_filter = radar_filter
        self.states       = states
        self.pos_data     = pos_data
        self.measured_positions  = []
        self.estimated_positions = []
        self.nees = []

    def gen_data_set(self):
        rs, thetas, phis = self.radar.gen_data(self.pos_data)
        noisy_rs, noisy_thetas, noisy_phis = self.radar.sense(rs, thetas, phis)
        xs, ys, zs = self.radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)
        self.measured_values    = np.array(list(zip(noisy_rs,noisy_thetas,noisy_phis)))
        self.measured_positions = np.array(list(zip(xs,ys,zs)))

    def process_filter(self,with_nees = False):
        # Initialization of the lists of:
        # estimated states: results of the estimator on state space vector
        # nees: Normalized Estimated Error Squared (if mode triggered)
        est_states, nees = [],[]

        # Scrolling through the measurements
        for i,measurement in enumerate(self.measured_values):
            self.radar_filter.predict()
            self.radar_filter.update(measurement)
            est_states.append(self.radar_filter.x)

            if with_nees:
                # Computation of the error between true and estimated states
                states_tilde = np.subtract(est_states[i],np.reshape(self.states[i,:],(-8,1)))
                nees.append(*(states_tilde.T@inv(self.radar_filter.P)@states_tilde))

        # Conversion in arrays
        est_states = np.array(est_states)
        nees       = np.array(nees)
        # Extraction of the position (for plotting)
        est_xs     = est_states[:,0]
        est_ys     = est_states[:,3]
        est_zs     = est_states[:,6]
        self.estimated_positions = np.concatenate((est_xs,est_ys,est_zs),axis=1)
        self.nees = nees

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
            plt.plot(self.nees)

        plt.show()

    def launch_benchmark(self, with_nees = False):
        self.gen_data_set()
        self.process_filter(with_nees = with_nees)
        self.plot()
