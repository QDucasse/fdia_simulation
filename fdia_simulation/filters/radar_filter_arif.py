# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:46:27 2019

@author: qde
"""
import sympy
import numpy as np
from sympy.abc       import x, y, z
from sympy           import symbols, Matrix
from math            import sqrt, atan2
from fdia_simulation.filters.radar_filter_model import RadarModel
# Other model possibility
    # Following [ARIF2017] paper
    # X_next      = np.zeros((6,1))
    # X_next[0,0] = x / sqrt(x**2 + y**2 + z**2) - y - x * z / sqrt(x**2 + y**2)
    # X_next[1,0] = y / sqrt(x**2 + y**2 + z**2) - y - y * z / sqrt(x**2 + y**2)
    # X_next[2,0] = z / sqrt(x**2 + y**2 + z**2) + sqrt(x**2 + y**2)
    # X_next[3,0] = -2*y / sqrt(x**2 + y**2 + z**2) - 2*x*z / sqrt(x**2 + y**2 + z**2) / sqrt(x**2 + y**2) \
    #               -2*x + 2*y*z / sqrt(x**2 + y**2)
    # X_next[4,0] = -2*x / sqrt(x**2 + y**2 + z**2) - 2*y*z / sqrt(x**2 + y**2 + z**2) / sqrt(x**2 + y**2) \
    #               -2*y + 2*x*z / sqrt(x**2 + y**2)
    # X_next[5,0] = 2 * sqrt(x**2 + y**2) / sqrt(x**2 + y**2 + z**2)


if __name__ == "__main__":
    # ========================= [ARIF2017] model ===============================
    # For f (h is the same as above)
    fxu = Matrix([[x / sympy.sqrt(x**2 + y**2 + z**2) - y - x * z / sympy.sqrt(x**2 + y**2)],
                  [y / sympy.sqrt(x**2 + y**2 + z**2) - y - y * z / sympy.sqrt(x**2 + y**2)],
                  [z / sympy.sqrt(x**2 + y**2 + z**2) + sympy.sqrt(x**2 + y**2)],
                  [-2*y / sympy.sqrt(x**2 + y**2 + z**2) - 2*x*z / sympy.sqrt(x**2 + y**2 + z**2) / sympy.sqrt(x**2 + y**2) -2*x + 2*y*z / sympy.sqrt(x**2 + y**2)],
                  [-2*x / sympy.sqrt(x**2 + y**2 + z**2) - 2*y*z / sympy.sqrt(x**2 + y**2 + z**2) / sympy.sqrt(x**2 + y**2) -2*y + 2*x*z / sympy.sqrt(x**2 + y**2)],
                  [2 * sympy.sqrt(x**2 + y**2) / sympy.sqrt(x**2 + y**2 + z**2)]])
    fjac = fxu.jacobian(Matrix([x, y, z, vx, vy, vz]))
    # ==========================================================================
