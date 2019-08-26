# Noise finder results & Attack report

### Attack report

The results of several attacks is stored in `attack_report.txt`.
It contains the configuration that provided the results, the mean NEES, max NEES,
and number of anomalies detected for each filter observing the system.

### Launch
All filters are parameterized in the `examples/noise_finder_1model.py`. To
launch the tests, simply execute the file. The output results can be found
in the `results` folder under the name `noise_finder_results-date_time.csv`.

Whole testing campaign can take a while and control over `nb_iterations`
can help reduce the time but may increments the randomness of the
simulations.

### Format
CSV files are created this way:  
**Model - Configuration , Best value**  
For example:
* **CA-1Radar,2300.**
* **CT-2PRadars,3540**

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
