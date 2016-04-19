# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:32:06 2016

@author: jmwilson
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import rtree
from rtree import index
from geographiclib.geodesic import Geodesic

#==============================================================================
# Extended Magnitude range attenuation relationships for S-wave horizontal acceleration
#               a       b           c 1     c 2     d       e           sig
# PGA   rock    0.73    -7.2x10-4   1.16    0.96    -1.48   -0.42       0.31
#       soil    0.71    -2.38x10-3  1.72    0.96    -1.44   -2.45x10-2  0.33
# PGV   rock    0.86    -5.58x10-4  0.84    0.98    -1.37   -2.58       0.28
#       soil    0.89    -8.4x10-4   1.39    0.95    -1.47   -2.24       0.32
#==============================================================================
#R1 = np.sqrt(R**2+9)
#C = c1*exp(c2*(M-5))*(np.arctan(M-5)+np.pi/2.0)
#Y = 10**(a*M + b*(R1 + C) + d*np.log10(R1 + C) + e)

def C(M, c1, c2):
    return c1*exp(c2*(M-5))*(np.arctan(M-5)+np.pi/2.0)

def Y(R, M, motiontype):
    if motiontype = "PGA-rock":
        a = 0.73
        b = -7.2e-4
        c1 = 1.16
        c2 = 0.96
        d = -1.48
        e = -0.42
    elif motiontype = "PGA-soil":
        a = 0.71
        b = -2.38e-3
        c1 = 1.72
        c2 = 0.96
        d = -1.44
        e = -2.45e-2
    elif motiontype = "PGV-rock":
        a = 0.86
        b = -5.58e-4
        c1 = 0.84
        c2 = 0.98
        d = -1.37
        e = -2.58
    elif motiontype = "PGV-soil":
        a = 0.89
        b = -8.4e-4
        c1 = 1.39
        c2 = 0.95
        d = -1.47
        e = -2.24
    else:
        print "Motion Type not recognized\n"
        return 0
        
    return 10**(a*M + b*(np.sqrt(R**2+9) + C(M, c1, c2)) + d*np.log10(np.sqrt(R**2+9) + C(M, c1, c2)) + e)


xyz = open('../globalETAS/etas_outputs/etas_xyz.xyz', 'r')

lons = []
lats = []
etasVals = []

ETAS_array = []
GMPE_array = []

for line in xyz:
    line = line.split()
    ETAS_array.append([line[0], line[1], line[2]])
    GMPE_array.append([line[0], line[1], 0])
    if float(line[0]) not in lons:
        lons.append(float(line[0]))
    if float(line[1]) not in lats:
        lats.append(float(line[1]))
    etasVals.append(float(line[2]))

xyz.close()

ETAS_array = np.array(ETAS_array)
ETAS_rec = np.core.records.fromarrays(ETAS_array.transpose(), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])

GMPE_array = np.array(GMPE_array)
GMPE_rec = np.core.records.fromarrays(GMPE_array.transpose(), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])

t0 = time.time()

dists = {}
for line1 in ETAS_rec:
    lon1 = round(line1['x'],1)
    lat1 = round(line1['y'],1)
    M = '''TODO: Mapping from ETAS rate to source magnitude, we might need data for each region's background seismicity'''
    for i, line2 in enumerate(GMPE_rec):
        lon2 = round(line2['x'],1)
        lat2 = round(line2['y'],1)
        g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        distance = g['s12']
        
        S_Horiz_Soil_Acc = Y(distance, M, "PGA-soil")
        
        GMPE_rec[i]['z'] = max(GMPE_rec[i]['z'], S_Horiz_Soil_Acc)



t1 = time.time()
print t1 - t0


GMPE_rec.dump('GMPE_rec.p')




