# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:17:02 2019

@author: qde
"""

import numpy as np
from fdia_simulation.attackers import Attacker

class DOSAttacker(Attacker):
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

    + same than Attacker
    '''
    def __init__(self,mag = 1e4, *args, **kwargs):
        Attacker.__init__(self,*args,**kwargs)
        self.mag_vector = self.mag_vector*mag

    def attack_measurement(self, measurement):
        return Attacker.attack_measurement(self,measurement)

class DriftAttacker(Attacker):
    '''
    Implements an attack strategy consisting of injecting measurements to make
    one to three directional parameter drifting.
    Parameters
    ----------
    attack_drift:  float numpy array (3,1)
        Impact of the attack on the three position parameters.

    radar_pos: int
        Position of the attacked radar.
    '''
    def __init__(self, radar_pos, attack_drift = None, *args, **kwargs):
        if attack_drift is None:
            attack_drift = np.array([[0,0,10]]).T
        self.attack_drift = attack_drift
        self.radar_pos    = radar_pos
        Attacker.__init__(self,radar_pos = radar_pos,*args,**kwargs)

    def attack_measurement(self, measurement):
        '''
        Modifies the measurement following these three steps:
            - Computation of the position from the radar's measurements
            - Application of the attack_drift on the position.
            - Reconversion of the modified position in radar values.
        '''
        # Conversion of the radar values in the corresponding position
        r     = measurement[self.radar_pos*3    ,:]
        theta = measurement[self.radar_pos*3 + 1,:]
        phi   = measurement[self.radar_pos*3 + 2,:]
        x,y,z = self.radar.gen_position_vals(r,theta,phi)
        position = np.reshape(np.array([[x,y,z]]),(-2,1))

        # Applying the attack drift
        modified_position = position + self.attack_drift
        mod_x = modified_position[0,:]
        mod_y = modified_position[1,:]
        mod_z = modified_position[2,:]

        # Recomputing the radar values
        mod_r,mod_theta,mod_phi = self.radar.gen_radar_values(mod_x,mod_y,mod_z)
        modified_measurement = measurement
        modified_measurement[self.radar_pos*3    ,:] = mod_r
        modified_measurement[self.radar_pos*3 + 1,:] = mod_theta
        modified_measurement[self.radar_pos*3 + 2,:] = mod_phi

        return modified_measurement


class CumulativeDriftAttacker(DriftAttacker):
    '''
    Implements an attack strategy consisting of injecting measurements to build
    a drift delta_drift after delta_drift
    Parameters
    ----------
    delta_drift:  float numpy array (3,1)
        Impact of the attack on the three position parameters.

    radar_pos: int
        Position of the attacked radar.
    '''
    def __init__(self, delta_drift, *args, **kwargs):
        self.delta_drift = delta_drift
        DriftAttacker.__init__(self,attack_drift = delta_drift,*args,**kwargs)

    def attack_measurement(self, measurement):
        '''
        See DriftAttacker.attack_measurement() for more information
        The CumulativeDriftAttacker uses the same idea except it amplifies the drift
        with every step that goes by.
        Parameters
        ----------
        measurement: float numpy array
            Raw measurement coming from the attacked radar

        Returns
        -------
        modified_measurement: float numpy array
            Compromised measurement (added the cumulative drift)
        '''
        self.attack_drift += self.delta_drift
        return DriftAttacker.attack_measurement(self,measurement)
