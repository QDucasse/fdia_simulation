# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:31:41 2019

@author: qde
"""

from abc import ABC, abstractmethod
from filterpy.common import pretty_str
from numpy.random import randn


class Command(object):
    r'''Representation of a commanded parameter within a dynamic system. Used
    for modelization purposes only.

    Notes
    -----
    Commands here have nothing to do with the inputs of the filter. They are
    represented in order to simulate a data set modelizing changes in behavior
    (i.e. brutal acceleration, landing).
    '''
    def __init__(self,name,value,steps,delta):
        self.name  = name
        self.value = value
        self.steps = steps
        self.delta = delta

    def __eq__(self,other):
        '''
        Overloading of the equality by now saying two commands are equal if all
        their attributes are equal.
        Parameters
        ----------
        other: Command object
            The Command object that will be compared to the actual instance.

        Returns
        -------
        assertion: boolean
            Combination of the assertion of equality of all parameters.
        '''
        cond1 = (self.name == other.name)
        cond2 = (self.value == other.value)
        cond3 = (self.steps == other.steps)
        cond4 = (self.delta == other.delta)
        return cond1 & cond2 & cond3 & cond4

    def __repr__(self):
        return '\n'.join([
            'Command object',
            pretty_str('name', self.name),
            pretty_str('value', self.value),
            pretty_str('steps', self.steps),
            pretty_str('delta', self.delta)])


class NoisySensor(object):
    r'''Implements a noisy sensor.
    Parameters
    ----------
    std_noise: float
        Standard deviation of the measurement noise.

    Notes
    -----
    A NoisySensor will not generate any data itself but rather modify existing data
    to a noisy version of itself (closer to real life measurements).
    '''
    def __init__(self, std_noise=1.):
        self.std = std_noise

    def sense(self, val):
        '''
        Simulates real sensor by adding noise to a real value.
        Parameters
        ----------
        val: float
            Real value.

        Returns
        -------
        noisy_val: float
            Real value with simulated measurement noise.
        '''
        return (val + (randn()* self.std))

    def gen_sensor_data(self,val_list):
        '''
        Generates data of a noisy sensor with a given frequency.
        Parameters
        ---------
        val_list: float iterable
            List of measured values.

        t: float
            Total time of the measurements.

        time_std: float
            Standard deviation of the frequency of the measurements.

        Returns
        -------
        sensor_data: tuple list
            List of tuple composed of (time of the measurement, measurement)
        '''
        return [self.sense(val) for val in val_list]


class NoisyFrequencySensor(NoisySensor):
    r'''Implements a noisy sensor with a given frequency of data rate.
    Parameters
    ----------
    std_noise: float
        Standard deviation of the measurement noise.

    frequency: float
        Frequency of the data sampling.

    Notes
    -----
    The format of the data outputed by the NoisyFrequencySensor is no longer the same
    as NoisySensor. The time of the sampling is now included within the data list.
    '''
    def __init__(self,std_noise = 1., frequency = 1):
        if frequency == 0:
            raise ValueError("frequency can not be equal to 0")
        super().__init__(std_noise)
        self.frequency = frequency

    def gen_sensor_data(self,val_list,t,time_std):
        '''
        Generates data of a noisy sensor with a given frequency.
        Parameters
        ---------
        val_list: float iterable
            List of measured values.

        t: float
            Total time of the measurements.

        time_std: float
            Standard deviation of the frequency of the measurements.

        Returns
        -------
        sensor_data: tuple list
            List of tuple composed of (time of the measurement, measurement)
        '''
        # In case of a null time_std
        if time_std == None:
            time_std = self.frequency / 100

        sensor_data = [] # Output of the function
        dt = 0           # Time as it will be different depending on the sensor
        for i in range(t*self.frequency):
            dt    += 1/self.frequency
            t_i   =  dt + randn() * time_std   # Add noise to the time
            val_i =  np.sense(val_list[i])     # Add noise to the values
            sensor_data.append([t_i, val_i])

        return sensor_data



class MovingTarget(ABC):
    r'''Abstract class of a moving target model
    Parameters
    ----------
    command_list: Command list
        List of the commanded value of the target.
    '''

    def __init__(self,command_list):
        super().__init__()
        self.commands = {}
        for command in command_list:
            self.add_command(command)


    def add_command(self,command):
        '''
        Adds a command to the instance variable commands corresponding to a dictionary
        linking the names (key) and the commands themselves (value)
        '''
        self.commands[command.name] = command

    def change_command(self,name,value,steps):
        '''
        Changes the given parameter to a goal within a certain number of steps
        subclasses must implement a correct changing function for each command

        Parameters
        ----------
        name : string
            Name of the targetted command.

        value : float
            New value of the commanded parameter.

        steps : int
            Number of steps during which the commanded parameter should reach its
            new value.

        Notes
        -----
        The function change_command relies on the definition of a change_[commanded_parameter]
        for each command in the system. These definitions should be made in every subclass
        of MovingTarget.
        '''
        methodName = getattr(self,"change_"+name)
        methodName(value,steps)

    @abstractmethod
    def update(self):
        '''
        Implements the model that defines the dynamic system represented by a
        MovingTarget.
        '''
        pass
