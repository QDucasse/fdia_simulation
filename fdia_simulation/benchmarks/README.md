# Benchmarks

---

### Simulation Benchmark
The `Benchmark` object is the highest-level component of the project.
It wraps the other components and creates the environment in which
the simulation will be executed.  

(IMAGE NEEDED) Links between components wrapped by the benchmark

The simulation made by the benchmark object is processed through the
following steps:
* Data generation from the **Track/Radar**
* Processing of the filter's **Predict/Update** cycles over the generated measurements
* Addition of the **Attacker's input** in order to modify (part of) the incoming measurements
* Computation of a **Performance criteria**
* Plotting of the trajectory, performance and model probabilities (in case of IMM)

**Note:** *The `Benchmark` object has a common interface for both one,
multiple radars and different data rates radars*

---

### Process noise finder

The **process noise matrix Q** is often the result of a trial and error
process testing the best-fitting model with a bank of q to be tested. The
`ProcessNoiseFinder` object launches a set of benchmarks with a given
*filter model* and *radar(s)*.

The *process noise finder* can be found in the `examples` folder. For every
one of the four models (CA, CV, CT and TA) and three configurations
(one radar, two radars and two radars with different data rates), it
 iterates over the following qs:

```csv
    0.01, 0.02 ... 0.09
    0.1 ,  0.2 ... 0.9
    1   ,    2 ... 9
    10  ,   20 ... 4000

```

The results of the iterations can be found in the `results` folder under
the name `noise_finder_results-date_time.csv`. Due to the random behavior
of certain part of the simulation, each simulation is repeated a certain
number of times to reduce the randomness. This can be managed through
the `nb_iterations` parameters of the `NoiseFinder`.

*file example: noise_finder_results-01-08-2019_15-06.csv*
```csv
    CV-1Radar,4000.0
    CA-1Radar,3990.0
    CT-1Radar,3970.0
    TA-1Radar,1410.0
    CV-2Radars,3790.0
    CA-2Radars,3970.0
    CT-2Radars,3970.0
    TA-2Radars,70.0
    CV-2PRadars,3960.0
    CA-2PRadars,4000.0
    CT-2PRadars,3620.0
    TA-2PRadars,1010.0
```

---

### Examples of use

* Examples of use are present within the source code, simply execute
the files `examples/benchmark_1radar_imm3.py` or `examples/benchmark_2radars_imm3.py`
for a presentation of a simulated trajectory with no attackers.  
* The file `examples/benchmark_2period_radars_imm3.py` displays an example of an attack
on two radars with different data rates.  
* The file `examples/benchmark_template.py` provides all object definitions to design
your attack simulation.
* The file `examples/noise_finder_1model.py` provides a run of the `NoiseFinder` for
each of the model separately.
* The file `examples/mo_attacker_poc.py` provides an implementation of the attack described
in the research article [MO2010].
