# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:18:23 2019

@author: qde
"""

import warnings
import numpy as np
from numpy.random           import randn
from fdia_simulation.models import Radar

class BasicAttacker(object):
    '''
    Implements a basic attacker model.
    Parameters
    ----------
    filter: ExtendedKalmanFilter
        Filter of the attacked system.

    gamma: int numpy array
        Attack matrix: Diagonal square matrix with n = measurement (z) dimension.
        Terms on the diagonal are either equal to 0 (no attack on this parameter)
        or 1 (attack on the parameter).
        Example: 2 radars
         np.array([[0, 0, 0, 0, 0, 0],  <--- No attack on r1
                   [0, 0, 0, 0, 0, 0],  <--- No attack on theta1
                   [0, 0, 0, 0, 0, 0],  <--- No attack on phi1
                   [0, 0, 0, 1, 0, 0],  <--- Attack on r2    )
                   [0, 0, 0, 0, 1, 0],  <--- Attack on theta2 } Attack on radar2
                   [0, 0, 0, 0, 0, 1]]) <--- Attack on phi2  )

    mag_vector: float numpy array
        Attack magnitude vector. The attack will consist of adding the quantity
        Gamma@mag_vector to the actual measurements.

    t0: float
        Time of beginning of the attack.

    time: int
        Duration of the attack (number of update steps)

    Attributes
    ----------
    Same as parameters +
    dim_z: int
        Dimension of the measurements.

    current_time: int
        Progression of the attack (from t0 to time)
    '''
    def __init__(self, filter, t0, time,
                 gamma = None, mag_vector = None, radar_pos = None):
        # Store the filter and its dimension
        self.filter = filter
        self.dim_z  = filter.dim_z
        # print('dim_z = {0}'.format(dim_z))

        # If gamma is not specified but the attacked radar position (in the
        # measurement matrix) is
        if (gamma is None) and not(radar_pos is None):
            gamma = np.zeros((self.dim_z,self.dim_z))
            gamma[radar_pos*3  , radar_pos*3]   = 1 # Attacked r
            gamma[radar_pos*3+1, radar_pos*3+1] = 1 # Attacked theta
            gamma[radar_pos*3+2, radar_pos*3+2] = 1 # Attacked phi

        # If the magnitude vector is not specified but the attacked radar position
        # (in the measurement matrix) is
        if (mag_vector is None) and not(radar_pos is None):
            mag_vector = np.zeros((self.dim_z,1))
            mag_vector[radar_pos*3    , 0] = 1
            mag_vector[radar_pos*3 + 1, 0] = 1
            mag_vector[radar_pos*3 + 2, 0] = 1

        # The attack matrix should be a squared matrix with n = dim_z
        if np.shape(gamma) != (self.dim_z,self.dim_z):
            raise ValueError('Gamma should be a square matrix with n=dim_z')

        if np.shape(mag_vector) != (self.dim_z,1):
            msg = """The magnitude vector should have the following dimensions:
                     (1,dim_z) with dim_z = {0} here""".format(self.dim_z)
            raise ValueError(msg)


        self.gamma       = gamma
        self.mag_vector  = mag_vector

        # If the attackers input is null, the attack will have no effect
        if np.array_equal(self.gamma@self.mag_vector,np.zeros((self.dim_z,1))):
            msg = """With the current attack matrix (gamma) and magnitude vector,
                     your attack will have no effect"""
            warnings.warn(msg,Warning)

        # Time attributes
        self.t0           = t0
        self.time         = time
        self.current_time = 0

    def attack_measurements(self, z):
        '''
        Performs the attack on a given measurement. It simply consists of adding
        to that measurement the quantity gamma@mag_vector. This means you are
        specifying which parameters are modified in gamma (which radar is attacked)
        and with what magnitude in the magnitude vector.
        Parameters
        ----------
        z: float numpy array (dim_z,1)
            Measurement as outputed by the radar under attack.

        Returns
        -------
        modified_z: float numpy array (dim_z,1)
            Modified measurement consisting of:
            Initial measurement + Gamma * Magnitude vector
        '''
        modified_z = z + self.gamma@self.mag_vector
        return modified_z

    def listen_measurements(self,z):
        '''
        Monitors the duration (beginning and end) of the attack.
        Parameters
        ----------
        z: float numpy array
            Measurement
        '''
        measurement       = z
        beginning_reached = self.t0 <= self.current_time
        end_reached       = (self.current_time - self.t0) >= self.time
        if beginning_reached and not(end_reached):
            measurement = self.attack_measurements(z)
        self.current_time += 1
        return measurement

class BruteForceAttacker(BasicAttacker):
    '''
    Implements an attack strategy consisting of systematically giving out wrong
    measurements in order to make the filter condemn the attacked sensor. This
    attack is not stealthy and can have little to no impact if there are several
    other sensors observing the system.
    Parameters
    ----------
    mag: float
        The magnitude added to the magnitude vector in order to make it always be
        detected by the fault detector.

    + same than BasicAttacker
    '''
    def __init__(self,mag = 1e6, *args, **kwargs):
        BasicAttacker.__init__(self,*args,**kwargs)
        self.mag_vector = self.mag_vector*mag

    def attack_measurements(self, z):
        return BasicAttacker.attack_measurements(self,z)

class DriftAttacker(BasicAttacker):
    '''
    Implements an attack strategy consisting of injecting measurements to make
    one to three directional parameter drifting.
    Parameters
    ----------
    attack_drift:  float numpy array (3,1)
        Impact of the attack on the three position parameters.

    radar: Radar object
        Attacked radar.
    '''
    def __init__(self, radar, radar_pos, attack_drift = None, *args, **kwargs):
        if attack_drift is None:
            attack_drift = np.array([[0,0,10]]).T
        self.attack_drift = attack_drift
        self.radar        = radar
        self.radar_pos    = radar_pos
        BasicAttacker.__init__(self,radar_pos = radar_pos,*args,**kwargs)

    def attack_measurements(self, z):
        '''
        Modifies the measurement following these three steps:
            - Computation of the computed position from the radar's measurements
            - Application of the attack_drift on the position.
            - Reconversion of the modified position in radar values.
        '''
        # Conversion of the radar values in the corresponding position
        r     = z[self.radar_pos*3    ,:]
        theta = z[self.radar_pos*3 + 1,:]
        phi   = z[self.radar_pos*3 + 2,:]
        x,y,z = self.radar.gen_position_vals(r,theta,phi)
        position = np.reshape(np.array([[x,y,z]]),(-2,1))

        # Applying the attack drift
        modified_position = position + self.attack_drift
        mod_x = modified_position[0,:]
        mod_y = modified_position[1,:]
        mod_z = modified_position[2,:]

        # Recomputing the radar values
        mod_r,mod_theta,mod_phi = self.radar.gen_radar_values(mod_x,mod_y,mod_z)
        modified_z = np.zeros((self.dim_z,1))
        modified_z[self.radar_pos*3    ,:] = mod_r
        modified_z[self.radar_pos*3 + 1,:] = mod_theta
        modified_z[self.radar_pos*3 + 2,:] = mod_phi

        return modified_z


if __name__ == "__main__":
    unittest.main()
