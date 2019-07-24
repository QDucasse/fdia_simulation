fdia_simulation - False Data Injection Attack on a simulated Air Traffic Controller
-----------------------------------------------------------------------------------------

Installation
------------

For downloading and installing the source code of the project:

::

    cd <directory you want to install to>
    git clone http://github.com/rlabbe/filterpy
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


License
-------
MIT License (MIT)

Copyright (c) 2019 Quentin Ducasse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
