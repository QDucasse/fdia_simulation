fdia_simulation - False Data Injection Attack on a simulated Air Traffic Controller
-----------------------------------------------------------------------------------------

Installation
------------

For downloading and installing the source code of the project:

::

    cd <directory you want to install to>
    git clone https://pan.kereval.com/qde/fdia_simulation
    python setup.py install

Basic use
---------

WIP


Objectives and Milestones of the project
----------------------------------------

Requirements
------------
**NOTE** This project was created with Python 3.7.3 and no backward compatibility is ensured.

This project uses NumPy, SciPy, Matplotlib, and Python as base modules.
FilterPy zas the starting point and is therefore required as well.
SymPy is used only in precise cases and may not be mandatory for you to install.

If installed as specified, the requirements are stated in the ``requirements.txt`` file
and therefore automatically installed.

Testing
-------

All tests are written to work with nose and/or pytest. Just type ``pytest`` or
``nosetests`` as a command line in the project.

The tests are not robust as they verify the integrity of the data generated but
not its quality. Testing the reaction of a bad designed filtered on a very demanding
trajectory was not the point of the project. However, many examples allow you to
try and test the results of many different combinations.

References
----------

I used Roger Labbe Jr. "Kalman and Bayesian Filters in Python" as a starting point
and not only did it showed me the FilterPy library but it also made me discover
Bar Shalom "Estimation with Application to Tracking and Navigation"
