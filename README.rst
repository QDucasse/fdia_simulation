fdia_simulation - False Data Injection Attack on a simulated Air Traffic Controller
-----------------------------------------------------------------------------------------

Installation
------------

I worked on the project through a virtual environment with ``virtualenvwrapper``
For downloading and installing the source code of the project:

::

    cd <directory you want to install to>
    git clone https://pan.kereval.com/qde/fdia_simulation
    python setup.py install

Basic use
---------

Explanation of the situation (graph + state/model)
::
    from fdia_simulation.models     import Radar, Track
    from fdia_simulation.filters    import RadarFilterCA
    from fdia_simulation.benchmarks import Benchmark

    # Creation of a trajectory
    trajectory = Track()
    states = trajectory.gen_takeoff() # Takeoff trajectory here
    x0=states[0,0] # Initial state that will be passed to the filter
    y0=states[0,3]
    z0=states[0,6]

    # Creation of a radar observing the trajectory
    radar = Radar(x=0,y=500)

    # Creation of the filter that will estimate the position of the plane given
    # the radar's measurements
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 3070.,
                                    x0 = x0, y0 = y0, z0 = z0, radar = radar)
    # Here, the model chosen for the filter is CA (Constant Acceleration)

    # Creation and launching of the benchmark doing:
    # - Data generation from the radar
    # - Filter processing
    # - Attacker influence
    # - Performance computation
    # - Plotting
    benchmark_ca = Benchmark(radars = radar, radar_filter = radar_filter_ca,states = states)
    benchmark_ca.launch_benchmark(with_nees = True)

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

All 500~ tests are written to work with nose and/or pytest. Just type ``pytest`` or
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
