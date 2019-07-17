# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:31:41 2019

@author: qde
"""
from abc             import ABC, abstractmethod
from filterpy.common import pretty_str

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
