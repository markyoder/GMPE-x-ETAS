# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:32:06 2016

@author: jmwilson
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt

xyz = open('../globalETAS/etas_outputs/etas_xyz.xyz')

lons = []
lats = []
etasVals = []
ETAS_array = []

for line in xyz:
    line = line.split()
    ETAS_array.append([line[0], line[1], line[2]])
    if float(line[0]) not in lons:
        lons.append(float(line[0]))
    if float(line[1]) not in lats:
        lats.append(float(line[1]))
    etasVals.append(float(line[2]))


ETAS_array = np.array(ETAS_array)
ETAS_array = np.core.records.fromarrays(ETAS_array.transpose(), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])

#==============================================================================
# etas_gid = []
# for j, lat in enumerate(lats):
#     etas_grid.append([])
#     for i, lon in enumerate(lons):
#         etas_grid[j].append([lon, lat, etasVals])
#==============================================================================













#==============================================================================
# Extended Magnitude range attenuation relationships
#               a       b           c 1     c 2     d       e           sig
# PGA   rock    0.73    -7.2x10-4   1.16    0.96    -1.48   -0.42       0.31
#       soil    0.71    -2.38x10-3  1.72    0.96    -1.44   -2.45x10-2  0.33
# PGV   rock    0.86    -5.58x10-4  0.84    0.98    -1.37   -2.58       0.28
#       soil    0.89    -8.4x10-4   1.39    0.95    -1.47   -2.24       0.32
#==============================================================================
R1 = np.sqrt(R**2+9)
C = c1*exp(c2*(M-5))*(np.arctan(M-5)+np.pi/2.0)
Y = 10**(a*M + b*(R1 + C) + d*np.log10(R1 + C) + e)



#plt.imshow(ETAS_array.z)