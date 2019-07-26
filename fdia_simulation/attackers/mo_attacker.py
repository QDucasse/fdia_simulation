# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:45:28 2019

@author: qde
"""

import numpy as np
from numpy.linalg              import inv, norm
from scipy.linalg              import solve_discrete_are,lstsq
from numpy.random              import randn
from filterpy.common           import pretty_str
from filterpy.kalman           import KalmanFilter


class UnstableData(object):
    r'''Implements a model for storing unstable data.
    Parameters
    ----------
    val: float
        Unstable eigenvalue.

    vect: numpy float array
        Unstable eigenvector linked to the previous eigenvalue.

    position: int
        Position of the eigenvalue in the studied matrix.
    '''
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


class MoAttacker(object):
    r''' Implements an attack simulation based on the research article Mo et al.
    2010 in order to generate a sequence of false measurements for a given subset
    of sensors.
    Parameters
    ----------
    kf: KalmanFilter
        Estimator of the system under attack.

    Attributes
    ----------
    unst_data: UnstableData list
        List of UnstableData objects (eigenvalue, eigenvector and position of the
        eigenvalue).

    attack_sequence: float numpy array
        Column-stacked array as np.array([[ y0 | y1 | ... | y_attsize ]])
        corresponding to the values the attacker will have to inject in the
        system to compromise it.
    '''
    def __init__(self,kf,fb = False):
        super().__init__()
        self.unst_data = []
        self.attack_sequence = np.zeros((kf.dim_z,0))
        self.kf = kf
        self.fb = fb

    def add_false_measurement(self,yai):
        '''
        Adds the new computed false measurement to the sequence by adding the column
        to the right.
        Parameters
        ----------
        yai: float numpy array
            New vector taking part of the attack sequence for time i.
        '''
        self.attack_sequence = np.hstack([self.attack_sequence, yai])


    def compute_unstable_eig(self,A):
        '''
        Looks up the unstable eigenvectors and eigenvalues from the input matrix.
        Parameters
        ----------
        A: float matrix
            The matrix which eigenvectors and values will be tested.

        Returns
        -------
        unst_data: UnstableData iterable
            List of UnstableData (eigenvalue, eigenvector and position of the value in the matrix).

        Notes
        -----
        The list of unstable eigenvectors and eigenvalues is not only returned but
        also directly added to the class attribute.
        '''
        eig_vals, eig_vects = np.linalg.eig(A)
        unst_data = []
        # Unstable eigenvalues are aigenvalues having their real part greater than 0
        for counter,value in enumerate(eig_vals):
            if np.real(value) >= 0:
                unst_data.append(UnstableData(value, eig_vects[:,counter],counter))
        self.unst_data = unst_data
        return unst_data

    def compute_steady_state_P(self):
        '''
        Computes the process to get P (covariance matrix on state) in steady
        state as the solution of a discrete Ricatti equation.

        Returns
        -------
        ss_P: matrix
            Covariance matrix on state when the system is on steady state.

        Notes
        -----
        This function should be used if the system is equiped with some kind of
        feedback linearization.
        '''
        ss_P = solve_discrete_are(self.kf.F.T, self.kf.H.T, self.kf.Q, self.kf.R)
        return ss_P

    def compute_steady_state_K(self):
        '''
        Computes the process to get K (Kalman gain) in steady state as the
        solution of a discrete Ricatti equation.
        Returns
        -------
        ss_K: matrix
            Kalman gain matrix when the system is on steady state.
            K = P*H'*inv(H*P*H'+R)
        '''
        kf = self.kf
        if self.fb: P = self.compute_steady_state_P()
        else: P = kf.P
        ss_K = P@(kf.H.T)@inv(kf.H@P@kf.H.T + kf.R)
        return ss_K

    def compute_attackers_input(self,ss_K,Gamma):
        '''
        Computes the initial space in which the attacker will have to find the
        initial steps of the attack sequence.
        Parameters
        ----------
        ss_K: float matrix
            Kalman gain of the estimator in steady state.

        Gamma: int matrix
            Attack matrix, saying which sensors have been compromised.

        Returns:
        --------
        attackers_input: float matrix
            Matrix
        '''
        kf = self.kf
        attackers_input = np.concatenate((-(kf.F - ss_K@kf.H@kf.F)@ss_K@Gamma, -ss_K@Gamma),axis=1)
        return attackers_input

    def initialize_attack_sequence(self,attackers_input,attack_vector):
        '''
        Computes the two first element of the attack sequence.
        Parameters
        ----------
        attackers_input: float matrix
            Matrix of the attacker's needed input to reach Cv.

        attack_vector: float numpy array
            Unstable eigenvector under which the attack is being made.

        Returns
        -------
        ya0, ya1: float numpy arrays
            First and second elements of the attack sequence.
        '''
        # Computes the first two elements by taking
        false_measurements,_,_,_ = lstsq(attackers_input, attack_vector)

        # Reshape correctly the result
        ya0 = false_measurements[0:self.kf.dim_z,0].reshape((self.kf.dim_z,1))
        ya1 = false_measurements[self.kf.dim_z:, 0].reshape((self.kf.dim_z,1))
        return ya0, ya1

    def compute_max_norm(self,Gamma,ya0,ya1):
        '''
        Computes the maximal norm after simulating the first two measurements.
        Parameters
        ----------
        Gamma: int matrix
            Attack matrix of the system.

        ya0, ya1: float numpy arrays
            Two first steps of the attack sequence.

        Returns
        -------
        M: float
            Maximal norm of the attack sequence.

        Notes
        -----
        - e corresponds to the error between the healthy and compromised system.
        - z corresponds to the cimulation of a received measurement by the compromised
        system (real measure + compromission of a subset of sensors)
        '''
        # e: error between healthy and compromised system
        kf = self.kf
        e = np.zeros((kf.dim_z,2))
        e[:,0] = (-kf.K@Gamma@ya0).squeeze()
        e[:,1] = (kf.F-kf.K@kf.H@kf.F)@(e[:,0].reshape((kf.dim_z,1))-kf.K@Gamma@ya1).squeeze()

        # z: simulation of the received measure by the plant (real measure + compromission)
        z = np.zeros((kf.dim_z,2))
        # First state is null so the received measure = attacker's input
        z[:,0] = (Gamma@ya0).squeeze()
        # Second state needs to be estimated (H*F*e) then added to the attacker's input
        z[:,1] = (kf.H@kf.F@e[:,0].reshape((kf.dim_z,1)) + Gamma@ya1).squeeze()

        M = max(norm(z[:,0]),norm(z[:,1]))

        return M

    def attack_parameters(self,value_position):
        '''
        Given a position value, returns the corresponding eigenvalue, eigenvector
        and attack matrix Gamma.
        Parameters
        ----------
        value_position: int
            Position of the targetted eigenvalue in the state transition matrix.

        Returns
        -------
        attack_val: float
            Eigen value under which the attack will happen.

        attack_vect: float numpy array
            Eigen vector under which the attack will happen.

        Gamma: int matrix
            Attack matrix: diagonal matrix where yij = 1 if the ith sensor is
            compromised and 0 if not.
        '''
        chosen_attack = self.unst_data[value_position]
        attack_val  = chosen_attack.value
        attack_vect = chosen_attack.vector.reshape((self.kf.dim_x,1))
        attack_pos  = chosen_attack.position

        Gamma  = np.zeros((self.kf.dim_z,self.kf.dim_z))
        Gamma[attack_pos][attack_pos] = 1

        return attack_val,attack_vect,Gamma


    def compute_attack_sequence(self, attack_size, pos_value = 0, logs=False):
        '''
        Creates the attack sequence (aka the falsified measurements passed to the filter).
        Parameters:
        -----------
        attack_size: int
            Duration of the attack (number of steps).

        pos_value: int
            Position of the unstable value along which the attacker should
            attack.

        logs: boolean
            Displays the logs of the different steps if True. Default value: False

        Returns
        -------
        attack_sequence: float numpy array
            Column-stacked array as np.array([[ y0 | y1 | ... | y_attsize ]])
            corresponding to the values the attacker will have to inject in the
            system to compromise it.

        Gamma: float numpy array
            Attack vector
        '''
        kf = self.kf
        # Unstable eigenvalues of A and associated eigenvectors
        self.compute_unstable_eig(kf.F)
        if not self.unst_data:
            # If the unstable list is empty, no attacks are possible
            return "No unstable values available for attack"

        else:
            # Choice of an eigenvalue under which the attack will be created
            attack_val,attack_vect,Gamma = self.attack_parameters(pos_value)

        if logs: print("Eigen value: \n{0}\n".format(attack_val))
        if logs: print("Eigen vector: \n{0}\n".format(attack_vect))
        if logs: print("Attack matrix: \n{0}\n".format(Gamma))

        # Attacker's input to reach v
        ss_K = self.compute_steady_state_K()
        if logs: print("Steady State K: \n{0}\n".format(ss_K))

        # Attacker input: -[(A-KCA)KGamma KGamma]
        attackers_input = self.compute_attackers_input(ss_K, Gamma)
        if logs: print("Attackers input: \n{0}\n".format(attackers_input))

        # Attack sequence initialization
        #1. Initialization of the false measurements
        ya0, ya1 = self.initialize_attack_sequence(attackers_input,attack_vect)
        if logs: print("First false measurements: \nya0:\n{0}\nya1:\n{1}\n".format(ya0,ya1))

        #2. Initialization of the first "real" measurements -> to determine the max norm
        M = self.compute_max_norm(Gamma,ya0,ya1)

        ya0 = ya0/M
        ya1 = ya1/M

        self.add_false_measurement(ya0)
        self.add_false_measurement(ya1)

        ystar = kf.H@attack_vect

        for i in range(2,attack_size):
            yai = self.attack_sequence[:,i-2].reshape((kf.dim_z,1)) - attack_val**(i-2)/M * ystar
            self.add_false_measurement(yai)

        if logs: print("Attack Sequence: \n{0}\n".format(self.attack_sequence))

        return self.attack_sequence, Gamma

    def attack_measurements(self):
        '''
        Alters the measurements with the attack sequence
        '''
        pass


class ExtendedMoAttacker(MoAttacker):

    def compute_steady_state_K(self):
        '''
        Computes the process to get K (Kalman gain) in steady state as the
        solution of a discrete Ricatti equation.
        Returns
        -------
        ss_K: matrix
            Kalman gain matrix when the system is on steady state.
            K = P*H'*inv(H*P*H'+R)
        '''
        kf = self.kf
        H = kf.HJacob(kf.x)
        if self.fb: P = self.compute_steady_state_P()
        else: P = kf.P
        ss_K = P@(H.T)@inv(H@P@H.T + kf.R)
        return ss_K

    def compute_attackers_input(self,ss_K,Gamma):
        '''
        Computes the initial space in which the attacker will have to find the
        initial steps of the attack sequence.
        Parameters
        ----------
        ss_K: float matrix
            Kalman gain of the estimator in steady state.

        Gamma: int matrix
            Attack matrix, saying which sensors have been compromised.

        Returns:
        --------
        attackers_input: float matrix
            Matrix
        '''
        kf = self.kf
        H =  kf.HJacob(kf.x)
        attackers_input = np.concatenate((-(kf.F - ss_K@H@kf.F)@ss_K@Gamma, -ss_K@Gamma),axis=1)
        return attackers_input

    def compute_max_norm(self,Gamma,ya0,ya1):
        '''
        Computes the maximal norm after simulating the first two measurements.
        Parameters
        ----------
        Gamma: int matrix
            Attack matrix of the system.

        ya0, ya1: float numpy arrays
            Two first steps of the attack sequence.

        Returns
        -------
        M: float
            Maximal norm of the attack sequence.

        Notes
        -----
        - e corresponds to the error between the healthy and compromised system.
        - z corresponds to the cimulation of a received measurement by the compromised
        system (real measure + compromission of a subset of sensors)
        '''
        # e: error between healthy and compromised system
        kf = self.kf
        H = kf.HJacob(kf.x)
        e = np.zeros((kf.dim_z,2))
        e[:,0] = (-kf.K@Gamma@ya0).squeeze()
        e[:,1] = (kf.F-kf.K@H@kf.F)@(e[:,0].reshape((kf.dim_z,1))-kf.K@Gamma@ya1).squeeze()

        # z: simulation of the received measure by the plant (real measure + compromission)
        z = np.zeros((kf.dim_z,2))
        # First state is null so the received measure = attacker's input
        z[:,0] = (Gamma@ya0).squeeze()
        # Second state needs to be estimated (H*F*e) then added to the attacker's input
        z[:,1] = (H@kf.F@e[:,0].reshape((kf.dim_z,1)) + Gamma@ya1).squeeze()

        M = max(norm(z[:,0]),norm(z[:,1]))

        return M

    def compute_attack_sequence(self, attack_size, pos_value = 0, logs=False):
        '''
        Creates the attack sequence (aka the falsified measurements passed to the filter).
        Parameters:
        -----------
        attack_size: int
            Duration of the attack (number of steps).

        pos_value: int
            Position of the unstable value along which the attacker should
            attack.

        logs: boolean
            Displays the logs of the different steps if True. Default value: False

        Returns
        -------
        attack_sequence: float numpy array
            Column-stacked array as np.array([[ y0 | y1 | ... | y_attsize ]])
            corresponding to the values the attacker will have to inject in the
            system to compromise it.

        Gamma: float numpy array
            Attack vector
        '''
        kf = self.kf
        H = kf.HJacob(kf.x)
        # Unstable eigenvalues of A and associated eigenvectors
        self.compute_unstable_eig(kf.F)
        if not self.unst_data:
            # If the unstable list is empty, no attacks are possible
            return "No unstable values available for attack"

        else:
            # Choice of an eigenvalue under which the attack will be created
            attack_val,attack_vect,Gamma = self.attack_parameters(pos_value)

        if logs: print("Eigen value: \n{0}\n".format(attack_val))
        if logs: print("Eigen vector: \n{0}\n".format(attack_vect))
        if logs: print("Attack matrix: \n{0}\n".format(Gamma))

        # Attacker's input to reach v
        ss_K = self.compute_steady_state_K()
        if logs: print("Steady State K: \n{0}\n".format(ss_K))

        # Attacker input: -[(A-KCA)KGamma KGamma]
        attackers_input = self.compute_attackers_input(ss_K, Gamma)
        if logs: print("Attackers input: \n{0}\n".format(attackers_input))

        # Attack sequence initialization
        #1. Initialization of the false measurements
        ya0, ya1 = self.initialize_attack_sequence(attackers_input,attack_vect)
        if logs: print("First false measurements: \nya0:\n{0}\nya1:\n{1}\n".format(ya0,ya1))

        #2. Initialization of the first "real" measurements -> to determine the max norm
        M = self.compute_max_norm(Gamma,ya0,ya1)

        ya0 = ya0/M
        ya1 = ya1/M

        self.add_false_measurement(ya0)
        self.add_false_measurement(ya1)

        ystar = H@attack_vect

        for i in range(2,attack_size):
            yai = self.attack_sequence[:,i-2].reshape((kf.dim_z,1)) - attack_val**(i-2)/M * ystar
            self.add_false_measurement(yai)

        if logs: print("Attack Sequence: \n{0}\n".format(self.attack_sequence))

        return self.attack_sequence, Gamma


if __name__ == "__main__":
    #  ================== Filter and system generation ========================
    kf = KalmanFilter(dim_x=2,dim_z=2)
    kf.x      = [0.,0.]   # Initial values state-space is [ x xdot ]'
    kf.H      = np.array([[1.,0.],  # Observation matrix
                          [0.,1.]])
    kf.F      = np.array([[1., 1.],
                          [0., 1.]]) # State transition matrix
    kf.R      = np.eye(2)
    kf.Q      = np.eye(2)
    kf.P      = np.array([[1., 0.],
                          [0., 1.]])
    kf.B      = np.array([[0.5, 1.]]).T
    xs        = kf.x      # Initial values for the data generation
    zs        = [[kf.x[0],kf.x[1]]] # We are measuring both the position and velocity
    pos       = [kf.x[0]] # Initialization of the true position values
    vel       = [kf.x[1]] # Initialization of the true velocity values
    noise_std = 1.

    # ==========================================================================
    # ======== Noisy position measurements generation for 30 samples ===========
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

    # ==========================================================================
    # ========================= Attacker generation ============================
    mo_attacker = MoAttacker(kf,fb = False)
    ss_K = mo_attacker.compute_steady_state_K()
    kf.K = ss_K
    Gamma = np.array([[0., 0.],
                      [0., 1.]])
    attackers_input = mo_attacker.compute_attackers_input(ss_K,Gamma)
    attack_vector = np.array([[0.,1.]]).T
    ya0,ya1 = mo_attacker.initialize_attack_sequence(attackers_input,attack_vector)
    M = mo_attacker.compute_max_norm(Gamma,ya0,ya1)
    mo_attacker.compute_attack_sequence(attack_size = 30,pos_value = 0,logs = True)
    # ==========================================================================
