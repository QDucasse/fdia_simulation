# False Data Injection Attack on a simulated Air Traffic Controller

### Installation

I worked on the project through a virtual environment with `virtualenvwrapper`
and I highly recommend to do so as well. However, whether or not you are in a
virtual environment, the installation proceeds as follows:

* For downloading and installing the source code of the project:

  ```bash
    $ cd <directory you want to install to>
    $ git clone https://pan.kereval.com/qde/fdia_simulation
    $ python setup.py install
  ```
* For downloading and installing the source code of the project in a new virtual environment:  

  *Download of the source code & Creation of the virtual environment*
  ```bash
    $ cd <directory you want to install to>
    $ git clone https://pan.kereval.com/qde/fdia_simulation
    $ cd fdia_simulation
    $ mkvirtualenv -a . -r requirements.txt VIRTUALENV_NAME
  ```
  *Launch of the environment & installation of the project*
  ```bash
    $ workon VIRTUALENV_NAME
    $ pip install -e .
  ```

  *Launch of the basic GUI*
  ```bash
    $ python fdia_simulation/app.py
  ```
Note that the GUI does not contain all the features of the project but allows 
you getting familiar with the components and interactions between them.
---

### Structure of the project

Quick presentation of the different modules of the project:
* [**Models:**][models]
Dynamic systems models.
* [**Anomaly detectors:**][detectors]
Detectors of wrong values coming from the sensors.  
* [**Filters:**][filters]
State estimators with different models/combinations.
* [**Attackers:**][attackers]
Controllers of a sensor modifying its outputed values.
* [**Benchmarks:**][benchmarks]
Wrapper of the module through a simplified interface.
* [**Helpers:**][helpers]
Plotting and file writer tools.

The **examples** folder provides a handfull of *situations* made possible by this project
as well as a *benchmark template* if you want to try the different functionalities with
some guidelines.

**NOTE:** *more information can be obtained through the `README.md` of each module and
therefore simply by clicking on those modules.*

---

### Requirements

This project uses five main libraries:
* [`NumPy`][numpy] as the **array/numerical** handler
* [`SciPy`][scipy] as the handler for **matrix operations**
* [`Matplotlib`][matplotlib] as the **plot** handler
* [`FilterPy`][filterpy] as the starting point of **filter design/state estimation**  
* [`SymPy`][sympy] as a Jacobian matrix finder using symbolic calculus

If installed as specified above, the requirements are stated in the ``requirements.txt`` file
and therefore automatically installed.  
However, you can install each of them separately with the command:
```bash
  $ pip install <library>
```


**NOTE:** *This project was created with Python 3.7.3 and no backward compatibility is
ensured.*  

---

### Basic use

![alt-text][basic_use]

We will first need to import the essential components from the different packages:
* `models` from which we will extract our **track** and **radar**
* `filters` where from which we will extract our **system estimator**
* `benchmarks` from which we will extract the high-level **benchmark** object.

```python
  from fdia_simulation.models     import Radar, Track
  from fdia_simulation.filters    import RadarFilterCA
  from fdia_simulation.benchmarks import Benchmark
```

Then, we need to create and link those components
```python
  ## Trajectory Generation
  trajectory = Track()
  states = trajectory.gen_takeoff() # Takeoff trajectory here
  x0,y0,z0 = trajectory.initial_position()
  # Initial position that will be passed to the filter

  ## Radar Creation
  radar = Radar(x=0,y=500)

  ## Estimator Creation
  radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 3070.,
                                  x0 = x0, y0 = y0, z0 = z0, radar = radar)
  # Here, the model chosen for the filter is CA (Constant Acceleration)
```

Now that all our elements are instanciated, we need to run the cycle described
above. This operation is made by the benchmark object through a process of:
* Data generation from the **Track/Radar**
* Processing of the filter's **Predict/Update** cycles over the generated measurements
* Addition of the **Attacker's input** in order to modify (part of) the incoming measurements
* Computation of a **Performance criteria**
* Plotting of the trajectory, performance and model probabilities (in case of IMM)

```python
  ## Benchmark Creation
  benchmark_ca = Benchmark(radars = radar, radar_filter = radar_filter_ca,states = states)
  benchmark_ca.launch_benchmark(with_nees = True)
```

---

### Objectives and Milestones of the project

- [X] State generation for different trajectories (take off, landing, ...)
- [X] Measurements generation by radars from system states
- [X] Anomaly detection on simple systems
- [X] Anomaly detection in the ATC simulation
- [X] 4 filter models (CA, CV, CT and TA) for one radar
- [X] 4 filter models for multiple radars  
- [X] 4 filter models for multiple radars with different data rates   
- [X] IMM Estimator for one radar working in all cases
- [X] Attacker model for the three cases
- [X] Two attacker types (brute force and inducted drift) for the three cases
- [X] Benchmark wrapper for the three cases
- [X] Performance indicator for one and two radars with the same data rate
- [X] Performance indicator for radars with different data rates  
- [X] Process noise finder for one given model in the three cases
- [X] Process noise finder for an IMM in the three cases
- [X] Unit tests for all components
- [X] Documentation via docstrings/READMEs
- [X] Installation guide

---

### Testing

All 570~ tests are written to work with `nose` and/or `pytest`. Just type `pytest` or
`nosetests` as a command line in the project. Every test file can still be launched
by executing the testfile itself.
```bash
  $ python fdia_simulation/tests/chosentest.py
  $ pytest
  $ nosetests
```

The tests are not robust as they verify the integrity of the data generated but
not its quality. What that means is that even if the result of a filter might be 
considered bad as the estimation is not correct, his behavior is correct. 
Testing the reaction of a bad designed filtered on a very demanding trajectory is 
not the point of the project. However, many examples allow you to try and test the 
results of many different combinations and that is how the filter should be designed.

---

### References

I used Roger Labbe Jr. "Kalman and Bayesian Filters in Python" as a starting point
and not only did it showed me the FilterPy library but it also made me discover
Bar Shalom "Estimation with Application to Tracking and Navigation"


[models]:https://pan.kereval.com/qde/fdia_simulation/tree/master/fdia_simulation/models
[detectors]:https://pan.kereval.com/qde/fdia_simulation/tree/master/fdia_simulation/anomaly_detectors
[filters]:https://pan.kereval.com/qde/fdia_simulation/tree/master/fdia_simulation/filters
[attackers]:https://pan.kereval.com/qde/fdia_simulation/tree/master/fdia_simulation/attackers
[benchmarks]:https://pan.kereval.com/qde/fdia_simulation/tree/master/fdia_simulation/benchmarks
[helpers]:https://pan.kereval.com/qde/fdia_simulation/tree/master/fdia_simulation/helpers

[numpy]:https://github.com/numpy/numpy
[scipy]:https://github.com/scipy/scipy
[matplotlib]:https://github.com/matplotlib/matplotlib
[filterpy]:https://github.com/rlabbe/filterpy
[sympy]:https://github.com/sympy/sympy

[basic_use]:https://pan.kereval.com/qde/fdia_simulation/raw/master/images/basic_use.png "Basic use of the project"
