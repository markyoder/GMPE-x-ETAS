# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:32:06 2016

@author: jmwilson
@co-author: mark yoder
#
# comments (yoder): we'll need to code this for Python3 compatibility.
#    ... and i'd rewrite some of this to separate code from data.
"""

import numpy as np
import math
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import rtree
from rtree import index
from geographiclib.geodesic import Geodesic
#
# define the color-cycle for fancy plotting:
_colors =  mpl.rcParams['axes.color_cycle']
#
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
#
#motion_type_prams = {'PGA-rock':{'a':0.73, 'b':-7.2e-4, 'c1':1.16, 'c2':0.96, 'd':-1.48, 'e':-0.42}, ...}
# but i'm lazier than that too, and i want to minimize mistakes made by mistyping a variable, so let's code it...
motion_type_prams_lst_vars = ['a', 'b', 'c1', 'c2', 'd', 'e']
motion_type_prams_lst = {'PGA-rock':[0.73, -7.2e-4, 1.16, 0.96, -1.48, -0.42],
                     'PGA-soil':[0.71, -2.38e-3, 1.72, 0.96, -1.44, -2.45e-2],
                     'PGV-rock':[0.86, -5.58e-4, 0.84, 0.98, -1.37, -2.58],
                     'PGV-soil':[0.89, -8.4e-4, 1.39, 0.95, -1.47, -2.24]}
motion_type_prams = {key:{ky:vl for ky,vl in zip(motion_type_prams_lst_vars, vals)} for key,vals in motion_type_prams_lst.items()}
#print('mtp: ', motion_type_prams)

def f_Y(R,M, a=None, b=None, c1=None, c2=None, d=None, e=None, motion_type='PGA-soil'):
	# experimenting a bit with this quasi-recursive call structure. this approach might be slow, and maybe we should just separate this into
	# two separate functions.
	if motion_type!=None:
		return f_Y(R,M, motion_type=None, **motion_type_prams[motion_type])
	else:
		return 10**(a*M + b*(np.sqrt(R**2+9) + C(M, c1, c2)) + d*np.log10(np.sqrt(R**2+9) + C(M, c1, c2)) + e)

def C(M, c1, c2):
    return c1*np.exp(c2*(M-5))*(np.arctan(M-5)+np.pi/2.0)


# ... but let's do this a different way as well. let's separate data from code, so put this into a dictionary (which is then a global
# variable, but that's ok... we can use a couple of trick, then, to pull the variables out into function calls. we can use,
# __dict__.update(motion_type_prams[mt]) to set local variables, or we can calla funciton like f(**motion_type_prams[mt]).

def Y(R, M, motiontype):
    if motiontype == "PGA-rock":
        a = 0.73
        b = -7.2e-4
        c1 = 1.16
        c2 = 0.96
        d = -1.48
        e = -0.42
    elif motiontype == "PGA-soil":
        a = 0.71
        b = -2.38e-3
        c1 = 1.72
        c2 = 0.96
        d = -1.44
        e = -2.45e-2
    elif motiontype == "PGV-rock":
        a = 0.86
        b = -5.58e-4
        c1 = 0.84
        c2 = 0.98
        d = -1.37
        e = -2.58
    elif motiontype == "PGV-soil":
        a = 0.89
        b = -8.4e-4
        c1 = 1.39
        c2 = 0.95
        d = -1.47
        e = -2.24
    else:
        print("Motion Type not recognized\n")
        return 0
        
    return 10**(a*M + b*(np.sqrt(R**2+9) + C(M, c1, c2)) + d*np.log10(np.sqrt(R**2+9) + C(M, c1, c2)) + e)
#
def etas_to_GM(etas_src='../globalETAS/etas_outputs/etas_xyz.xyz', fname_out='GMPE_rec.p', motion_type='PGA-soil'):
	# "ETAS to Ground-Motion:
	#xyz = open('../globalETAS/etas_outputs/etas_xyz.xyz', 'r')
	#
	#
	print('open etas_src file: ', etas_src)
	#
	with open(etas_src, 'r') as xyz:
		# i don't know why, except maybe because it does handle exceptions/crashes before the file is closed, but this
		# "open()" syntax seems to be recommended over open(), close() implicit blocks.
		#
		lons = []
		lats = []
		etasVals = []
		#
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

		#xyz.close()
	#
	print('etas_src loaded. load data into arrays and process.')
	#
	# this syntax burns memory and time, which will become important later. instead of x.transopse(), we can use zip(*x)... unless
	# it is otherwise known that zip(*x) is slow.
	#ETAS_array = np.array(ETAS_array)
	#ETAS_rec = np.core.records.fromarrays(ETAS_array.transpose(), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	ETAS_rec = np.core.records.fromarrays(zip(*ETAS_array), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	#
	#GMPE_array = np.array(GMPE_array)
	#GMPE_rec = np.core.records.fromarrays(GMPE_array.transpose(), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	GMPE_rec = np.core.records.fromarrays(zip(*GMPE_array), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	#
	l_etas = len(ETAS_rec)
	#
	t0 = time.time()
	dists = {}
	# how do we parallelize this? i think first, we want to use iterools(), so we can break up the whole nested loop into one list.
	# we can avoid doing extra arithmetic by doing the round() operations first -- loop over [[round(x), round(y), z], ...]
	# and calc the source magnitude, again, in the first loop  over the ETAS array. at that point, all the rows are independent and
	# we can parse them out to processes.
	for j,line1 in enumerate(ETAS_rec):
		lon1 = round(line1['x'],1)
		lat1 = round(line1['y'],1)
		M = '''TODO: Mapping from ETAS rate to source magnitude, we might need data for each region's background seismicity'''
		M=2.0
		print('line: %d/%d' % (j,l_etas))
		#
		for i, line2 in enumerate(GMPE_rec):
			lon2 = round(line2['x'],1)
			lat2 = round(line2['y'],1)
			#
			# note: this can be an expensie operation. for speed optimization, i think there is a "use spherical solution" option
			#    (by default, this uses an iterative algorithm), or we can code in the exact spherical solution. globalETAS includes a
			#    pretty comprehensive handling of distances and distance types. for now, let's implement a simple version. cut-and-paste
			#    the spherical solution, and switch it out here (i think it's fair enough to say now that we'll need something simple and
			#    fast, so geographiclib is probably out of the question).
			#
			#spherical_dist(lon_lat_from=[0., 0.], lon_lat_to=[0.,0.], Rearth = 6378.1):
			distance = spherical_dist(lon_lat_from=[lon1, lat1], lon_lat_to=[lon2, lat2])
			#g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
			#distance = g['s12']
			#	
			#S_Horiz_Soil_Acc = Y(distance, M, "PGA-soil")
			S_Horiz_Soil_Acc = f_Y(distance, M, motion_type)
			#
			GMPE_rec[i]['z'] = max(GMPE_rec[i]['z'], S_Horiz_Soil_Acc)
	#
	t1 = time.time()
	print(t1 - t0)
	#
	GMPE_rec.dump(fname_out)


def spherical_dist(lon_lat_from=[0., 0.], lon_lat_to=[0.,0.], Rearth = 6378.1):
	# Geometric spherical distance formula...
	# displacement from inloc...
	# inloc is a vector [lon, lat]
	# return a vector [dLon, dLat] or [r, theta]
	# return distances in km.
	#
	# also, we need to get the proper spherical angular displacement (from the parallel)
	#
	#Rearth = 6378.1	# km
	deg2rad=2.0*math.pi/360.
	#
	# note: i usually use phi-> longitude, lambda -> latitude, but at some point i copied a source where this is
	# reversed. oops. so just switch them here.
	# phi: latitude
	# lon: longitude
	#
	#phif  = inloc[0]*deg2rad
	#lambf = inloc[1]*deg2rad
	#phis  = self.loc[0]*deg2rad
	#lambs = self.loc[1]*deg2rad
	
	phif  = lon_lat_to[1]*deg2rad
	lambf = lon_lat_to[0]*deg2rad
	phis  = lon_lat_from[1]*deg2rad
	lambs = lon_lat_from[0]*deg2rad
	#
	#print ('phif: ', phif)
	#print('lambf: ', lambf)
	#
	dphi = (phif - phis)
	dlambda = (lambf - lambs)
	#this one is supposed to be bulletproof:
	sighat3 = math.atan( math.sqrt((math.cos(phif)*math.sin(dlambda))**2.0 + (math.cos(phis)*math.sin(phif) - math.sin(phis)*math.cos(phif)*math.cos(dlambda))**2.0 ) / (math.sin(phis)*math.sin(phif) + math.cos(phis)*math.cos(phif)*math.cos(dlambda))  )
	R3 = Rearth * sighat3
	#
	return R3

# yoder: let's code this up for both command line and interactive use:
if __name__=='__main__':
	#
	kwds={}
	pargs=[]		# poositional arguments.
	for arg in sys.argv[1:]:
		# assume some mistakes might be made. fix them here.
		arg.replace(',', '')
		#
		# note: module name is the first argument.
		if '=' in arg:
			kwds.update(dict([arg.split('=')]))
		else:
			pargs+=[arg]
		#
	#
	# enforce float types:
	#kwds = {key:float(val) for key,val in kwds.items()}
	#pargs = [float(x) for x in pargs]
	#
	X=etas_to_GM(*pargs, **kwds)
else:
	plt.ion()
	pass
#




