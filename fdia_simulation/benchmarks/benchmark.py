# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:06:12 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from copy                      import deepcopy
from numpy.linalg              import inv
from filterpy.kalman           import IMMEstimator
from fdia_simulation.models    import Radar, FrequencyRadar, Track

class Benchmark(object):
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
    def __init__(self,radars,radar_filter,states):
        # Checks if there are multiple radars or simply one
        if isinstance(radars,(Radar,FrequencyRadar)):
            self.radars = [radars]
        else:
            self.radars = radars

        self.radar_filter = radar_filter
        self.states       = states
        xs,ys,zs          = states[:,0], states[:,3], states[:,6]
        self.pos_data     = np.array(list(zip(xs,ys,zs)))
        # Check if the filter is an IMM or not
        self.is_freq_radar = False
        if  isinstance(self.radars[0],FrequencyRadar):
            self.is_freq_radar = True
        self.filter_is_imm = False
        if type(self.radar_filter) == IMMEstimator:
            self.filter_is_imm = True
        # Actual values to be plotted
        self.measured_values     = []
        self.labeled_values      = []
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
        # The length of the sampled position is needed in case of multiple radars
        # with the same data rates for the concatenation.
        first_radar = self.radars[0]
        sampled_position_data = first_radar.sample_position_data(self.pos_data)
        self.measured_values = np.reshape(np.array([[]]),(len(sampled_position_data),0))


        for i,radar in enumerate(self.radars):
            sampled_position_data = radar.sample_position_data(self.pos_data)
            # Data generation for the radar
            rs, thetas, phis = radar.gen_data(sampled_position_data)
            # Addition of white noise
            noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
            # Conversion in positions (for plotting purposes)
            xs, ys, zs = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)
            # Addition of the measurements from the radar to the global measurements
            current_measured_values = np.array(list(zip(noisy_rs,noisy_thetas,noisy_phis)))
            # Addition of the computed positions to the
            self.measured_positions.append(np.array(list(zip(xs,ys,zs))))
            # print('measured positions ({1}): \n{0}\n'.format(self.measured_positions,np.shape(self.measured_positions)))

            # If the radars do not have different data rates, the measurement
            # vector consists of the concatenation of the different measurements
            if not self.is_freq_radar:
                self.measured_values = np.concatenate((self.measured_values,current_measured_values),axis=1)

            # If the radars have different data rates, the measurement vector
            # consists of labeled measurements
            else:
                current_labeled_measurement = radar.compute_measurements(self.pos_data)
                self.labeled_values += current_labeled_measurement

        # The labeled measrurements (in case of frequency radars) are sorted by time
        self.labeled_values = sorted(self.labeled_values)


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
            # Default values for FrequencyRadars
            if self.is_freq_radar:
                measurements = self.labeled_values
            # Default values for radars with the same data rates
            else:
                measurements = self.measured_values

        # Initialization of the lists of:
        # estimated states: results of the estimator on state space vector
        # nees: Normalized Estimated Error Squared (if mode triggered)
        est_states, nees, probs = [],[],[]
        # Scrolling through the measurements
        for i,measurement in enumerate(measurements):
            self.radar_filter.predict()
            self.radar_filter.update(measurement)
            # print("================================================================")
            # print(self.radar_filter)
            # print("================================================================")
            current_state = deepcopy(self.radar_filter.x)
            est_states.append(current_state)
            # print('Estimate states vector:\n{0}\n'.format(est_states))
            if with_nees:
                if not self.is_freq_radar:
                    # The corresponding real state to the estimated one
                    state_id = int(self.radars[0].step * i)
                else:
                    # The corresponding real state to the estimated one
                    state_id = int(measurement.time)
                # Computation of the error between true and estimated states
                states_tilde = np.subtract(est_states[i],np.reshape(self.states[state_id,:],(-8,1)))
                nees.append(*(states_tilde.T@inv(self.radar_filter.P)@states_tilde))
            if self.filter_is_imm:
                probs.append(self.radar_filter.mu)

        # Conversion in arrays
        est_states = np.array(est_states)
        nees       = np.array(nees)
        probs      = np.array(probs)

        # print('estimated states: \n{0}\n'.format(est_states))
        # print(np.shape(est_states))
        # Extraction of the position (for plotting)
        est_xs     = est_states[:,0,:]
        est_ys     = est_states[:,3,:]
        est_zs     = est_states[:,6,:]
        # print('estimated xs: \n{0}\n'.format(est_xs))
        # print('estimated ys: \n{0}\n'.format(self.est_ys))
        # print('estimated zs: \n{0}\n'.format(self.est_zs))

        self.estimated_positions = np.concatenate((est_xs,est_ys,est_zs),axis=1)
        # print('estimated positions: \n{0}\n'.format(self.estimated_positions))
        self.nees  = nees
        self.probs = probs

    def generate_plotting_labels(self):
        '''
        Extracts the names of radar filter(s) for plotting purposes.
        '''
        multiple_radars = len(self.radars) > 1
        # If the filter is an IMM, we need the name of each of the used models.
        if self.filter_is_imm:
            filter_models = self.radar_filter.filters
            for model in filter_models:
                # Model name creation (e.g.: Estimation-CV, Estimation-CA...)
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

        estimated_xs = self.estimated_positions[:,0]
        estimated_ys = self.estimated_positions[:,1]
        estimated_zs = self.estimated_positions[:,2]

        fig = plt.figure(1)
        plt.rc('font', family='serif')
        ax = fig.gca(projection='3d')
        ax.plot(real_xs, real_ys, real_zs, label='Real trajectory',color='k',linestyle='dashed')
        # Radar measurements
        for i,radar in enumerate(self.radars):
            # Measured values by the ith radar
            measured_xs = self.measured_positions[i][:,0]
            measured_ys = self.measured_positions[i][:,1]
            measured_zs = self.measured_positions[i][:,2]
            ax.scatter(measured_xs, measured_ys, measured_zs,
                       marker='o',alpha = 0.3, label = 'Radar n°'+str(i+1)+' measurements')

            # Radar position
            ax.scatter(radar.x,radar.y,radar.z, marker = 'x', label = 'Radar n°'+str(i+1))

        # Estimated positions
        ax.plot(estimated_xs, estimated_ys, estimated_zs, color='orange', label= 'Estimated trajectory')

        # Axis labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend()
        fig.show()

        # Plotting the Normalized Estimated Error Squared (NEES)
        if len(self.nees)>0:
            fig2 = plt.figure(2)
            plt.title('Normalized Estimation Error Squared (NEES)')
            plt.plot(self.nees)
            fig2.show()

        # Plotting the model probabilities
        if self.filter_is_imm:
            fig3 = plt.figure(3)
            for i in range(len(self.probs[0,:])):
                plt.plot(self.probs[:,i],label = self.radar_filters_names[i])
            plt.xlabel('Time')
            plt.ylabel('Model probability')
            plt.title('Probabilities of the different models')
            plt.legend()
            fig3.show()


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
