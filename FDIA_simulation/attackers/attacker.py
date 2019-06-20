# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:10:54 2019

@author: qde
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import solve_discrete_are,lstsq
from filterpy.common import kinematic_kf, pretty_str
from numpy.random import randn

class UnstableData(object):
    
    def __init__(self,val,vect,position):
        self.value    = val
        self.vector   = vect
        self.position = position 

    def __repr__(self):
        return '\n'.join([
            'UnstableData object',
            pretty_str('value', self.value),
            pretty_str('vector', self.vector),
            pretty_str('position', self.position)])


class Attacker(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def change_measurements(self):
        pass


class MoAttacker(Attacker):
    
    def __init__(self,kf):
        super().__init__()
        self.unst_data = []
        self.attack_sequence = np.zeros((kf.dim_z,0))
    
    def add_false_measurement(self,yai):
        '''
        Adds the new computed false measurement to the sequence by adding the column
        to the right.
        '''
        self.attack_sequence = np.hstack([self.attack_sequence, yai])
    
    
    def compute_unstable_eig(self,A):
        '''
        Looks up the unstable eigenvectors and eigenvalues from the input matrix
        '''
        eig_vals, eig_vects = np.linalg.eig(A)
        for counter,value in enumerate(eig_vals):
            if np.real(value) >= 0:
                self.unst_data.append(UnstableData(value, eig_vects[:,counter],counter))
        
    def compute_steady_state_P(self,kf):
        '''
        P in steady state is the solution of a Ricatti equation
        '''
        ss_P = solve_discrete_are(kf.F.T, kf.H.T, kf.Q, kf.R)
        return ss_P
    
    def compute_steady_state_K(self,kf,ss_P):
        '''
        K in steady state is: P*H.T*(H*P*H.T + R)
        '''
        ss_K = ss_P@kf.H.T@inv(kf.H@ss_P@kf.H.T + kf.R)
        return ss_K
    
    def compute_attackers_input(self,kf,ss_P,ss_K,Gamma):
        '''
        '''
        attackers_input = np.concatenate((-(kf.F - ss_K@kf.H@kf.F)@ss_K*Gamma, -ss_K@Gamma),axis=1)
        return attackers_input
        
    def initialize_attack_sequence(self,kf,attackers_input,attack_vector):
        '''
        '''
        false_measurements,_,_,_ = lstsq(attackers_input, attack_vector)
        ya0 = false_measurements[0:kf.dim_z,0].reshape((kf.dim_z,1))
        ya1 = false_measurements[kf.dim_z:, 0].reshape((kf.dim_z,1))
        return ya0, ya1
    
    def compute_max_norm(self,kf,Gamma,ya0,ya1):
        # Initialization of the first "real" measurements -> to determine the max norm
        # e: error between healthy and compromised systems
        
        # A CHANGER 
        e = np.zeros((kf.dim_z,2))
        e[:,0] = (-kf.K@Gamma@ya0).squeeze()
        e[:,1] = (kf.F-kf.K@kf.H@kf.F)@(e[:,0].reshape((kf.dim_z,1))-kf.K@Gamma@ya1).squeeze()
        
        # z: simulation of the received measure by the plant (real measure + compromission)
        z = np.zeros((kf.dim_z,2))
        z[:,0] = (Gamma@ya0).squeeze()
        z[:,1] = (kf.H@kf.F@e[:,0].reshape((kf.dim_z,1)) + Gamma@ya1).squeeze()

        M = max(norm(z[:,0]),norm(z[:,1]))
        
        return M
    
    def attack_parameters(self,kf,value_position):
        #! TODO Error si position trop grande ou en dehors du reste
        
        chosen_attack = self.unst_data[value_position]
        attack_val  = chosen_attack.value
        attack_vect = chosen_attack.vector.reshape((kf.dim_z,1))
        attack_pos  = chosen_attack.position
        
        Gamma  = np.zeros((kf.dim_z,kf.dim_z))
        Gamma[attack_pos][attack_pos] = 1
        
        return attack_val,attack_vect,Gamma
    
    
    def compute_attack_sequence(self,kf,attack_size,logs=False):
        '''
        Creates the attack sequence (aka the falsified measurements passed to the filter)
        '''
        # Unstable eigenvalues of A and associated eigenvectors
        self.compute_unstable_eig(kf.F)
        if not self.unst_data:
            # If the unstable list is empty, no attacks are possible
            return "No unstable values available for attack"
        
        else:
            # Choice of an eigenvalue under which the attack will be created
            attack_val,attack_vect,Gamma = self.attack_parameters(kf,0)
            
        # Attacker's input to reach v
        
        ss_P = self.compute_steady_state_P(kf)
        if logs: print("Steady State P: \n{0}\n".format(ss_P))
        ss_K = self.compute_steady_state_K(kf,ss_P)
        if logs: print("Steady State K: \n{0}\n".format(ss_K))
        
        # Attacker input: -[(A-KCA)KGamma KGamma]
        attackers_input = self.compute_attackers_input(kf, ss_P, ss_K, Gamma)
        if logs: print("Attackers input: \n{0}\n".format(attackers_input))
        
        # Attack sequence initialization
        #1. Initialization of the false measurements
        ya0, ya1 = self.initialize_attack_sequence(kf,attackers_input,attack_vect)
        if logs: print("First false measurements: \nya0:\n{0}\nya1:\n{1}\n".format(ya0,ya1))
           
        #2. Initialization of the first "real" measurements -> to determine the max norm
        M = self.compute_max_norm(kf,Gamma,ya0,ya1)
        
        ya0 = ya0/M
        ya1 = ya1/M
        
        self.add_false_measurement(ya0)
        self.add_false_measurement(ya1)
        
        ystar = kf.H@attack_vect
        
        for i in range(2,attack_size):
            yai = self.attack_sequence[:,i-2].reshape((kf.dim_z,1)) - attack_val**(i-2)/M * ystar
            self.add_false_measurement(yai)
        
        if logs: print("Attack Sequence: \n{0}\n".format(self.attack_sequence))

        return self.attack_sequence
    
    def change_measurements(self):
        '''
        Alters the measurements with the attack sequence
        '''
        pass
  

class YangAttacker(Attacker):
    def __init__(self):
        super().__init__()

    def compute_attack_sequence(self):
        r'''Creates the attack sequence (aka the falsified measurements passed to the filter)
        '''
        pass
    
    def change_measurements(self):
        r'''Alters the measurement with the attack sequence
        '''
        pass
    

    
if __name__ == "__main__":
    kf = kinematic_kf(dim=1,order=1,dt=1)
    kf.dim_z  = 2
    kf.H      = np.array([[1.,0.],
                          [0.,1.]])
    kf.R      = np.eye(2)
    kf.x      = [0.,2.]   # Initial value
    xs        = kf.x      # Initial values for the data generation
    zs        = [[kf.x[0],kf.x[1]]] # We are measuring both the position and velocity
    pos       = [kf.x[0]] # Initialization of the true position values
    vel       = [kf.x[1]] # Initialization of the true velocity values
    noise_std = 1.
    
    # Noisy position measurements generation for 30 samples
    for _ in range(30): 
        last_pos = xs[0]
        last_vel = xs[1]
        new_vel  = last_vel
        new_pos  = last_pos + last_vel
        xs       = [new_pos, new_vel]
        z        = [new_pos + (randn()*noise_std), new_vel + (randn()*noise_std)]
        zs.append(z)
        pos.append(new_pos)
        vel.append(new_vel)
    
    mo_attacker = MoAttacker(kf)
    ss_P = mo_attacker.compute_steady_state_P(kf)
    ss_K = mo_attacker.compute_steady_state_K(kf,ss_P)
    kf.K = ss_K
    Gamma = np.array([[0., 0.],
                      [0., 1.]])
    attackers_input = mo_attacker.compute_attackers_input(kf,ss_P,ss_K,Gamma)
    attack_vector = np.array([[0.,1.]]).T
    ya0,ya1 = mo_attacker.initialize_attack_sequence(kf,attackers_input,attack_vector)
    M = mo_attacker.compute_max_norm(kf,Gamma,ya0,ya1)
    mo_attacker.compute_attack_sequence(kf,30,logs = True)
    
    
    
    
    
    
    
    