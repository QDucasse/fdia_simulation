# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:59:32 2019

@author: qde
"""

import unittest
import numpy as np
from nose.tools              import raises
from fdia_simulation.models  import Radar
from fdia_simulation.filters import RadarModel, RadarFilterCA,RadarFilterCV, MultipleRadarsFilterCA,MultipleRadarsFilterCV

class RadarModelTestCase(unittest.TestCase):
    @raises(TypeError)
    def test_no_initialization(self):
        abstractClassInstance = RadarModel()


if __name__ == "__main__":
    unittest.main()
