# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:32:06 2016

@author: jmwilson
@co-author: mark yoder
#
# comments (yoder): we'll need to code this for Python3 compatibility.
#    ... and i'd rewrite some of this to separate code from data.
"""
#
import numpy as np
import scipy
numpy = np		# i always write "numpy"; let's just catch it here.
import math
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import multiprocessing as mpp
import os
from scipy.interpolate import RectSphereBivariateSpline
from scipy.interpolate import RectBivariateSpline
#
import rtree
from rtree import index
from geographiclib.geodesic import Geodesic
#
# define the color-cycle for fancy plotting:
#_colors =  mpl.rcParams['axes.color_cycle']
_colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
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
#
#
def etas_to_mag(z_etas, area=100, t0=0., t1=0., t2=0., p=1.05, dm=1.0, mc=2.5):
	# m = log(N) + dm + mc
	#
	return numpy.log10((z_etas*area/(1.-p))*((t0+t2)**(1.-p) - (t0+t1)**(1.-p))) + dm + mc
#
def etas_to_GM(etas_src='etas_src/etas_japan2016_20160415_2300CDT_kml_xyz.xyz', fname_out='GMPE_rec.p', motion_type='PGA-soil', etas_size=None, gmp_size=None, n_procs=None):
	# "ETAS to Ground-Motion:
	# etas_size: if None, use raw data as they are. otherwise, re-size the lattice using scipy interpolation tools (grid_data() i think)
	# gmp_size: if None, use raw (etas) data size, otherwise, create a grid... and for these two variables, we need to decide if we want
	# to define the grid-size, or n_x/n_y, or have options for either. probably, handle a single number as a grid-size (and assume square);
	# handle an array as lattice dimensions.
	#
	#xyz = open('../globalETAS/etas_outputs/etas_xyz.xyz', 'r')
	n_procs=(n_procs or mpp.cpu_count())	
	#
	print('open etas_src file: ', etas_src)
	#
	ETAS_array = open_xyz_file(etas_src)
	ETAS_array['z']=numpy.log(ETAS_array['z'])
	#
	lons = sorted(list(set([x for x,y,z in ETAS_array])))
	lats = sorted(list(set([y for x,y,z in ETAS_array])))
	#
	######################################
	# resize/interpolate?::
	# 
	# are we re-sizing the lattice? for most cases, we can probably use a cartesian approximation, but scipy happens to have a
	# cartesian-spherical (aka, lat/lon) interpolator:
	# scipy.interpolate.RectSphereBivariateSpline... which breaks for complex data sets, like ETAS maps.
	# however, scipy.interpolate.interp2d works well.
	#
	# interpolate_scipy() handles the size variable as follows: list-like are interpreted as the new shape;
	# scalar-like are interpreted as a factor, new_size=(size*size[0], size*size[1])
	#if isinstance(etas_size,int) or isinstance(etas_size,float): etas_size=(etas_size, etas_size)
	#if isinstance(gmp_size,int) or isinstance(gmp_size,float): gmp_size=(gmp_size, gmp_size)
	#
	#
	if etas_size == None: etas_size=(len(lons), len(lats))
	if gmp_size  == None: gmp_size=etas_size
	#
	print('ETAS_array, size: {}, shape: {}/{}'.format(ETAS_array.size, ETAS_array.shape, etas_size))
	if not tuple(etas_size)==tuple(ETAS_array.shape):
		# we're resizing and interpolating.
		# note: i think an easier way to interpolate is to use PIL. load data into an "image" object img and use img.thumbnail(sz, opts...)
		ETAS_array = interpolate_scipy(ETAS_array, etas_size, fignum=0)
		#print('New ETAS_array, size: {}, shape: {}'.format(ETAS_array.size, ETAS_array.shape))
		#
		lons = sorted(list(set([x for x,y,z in ETAS_array])))
		lats = sorted(list(set([y for x,y,z in ETAS_array])))
	if isinstance(gmp_size,int) or isinstance(gmp_size,float): gmp_size=[gmp_size*x for x in etas_size]
	print('GMP_size: {}/{}'.format(gmp_size, etas_size))
	#
	ETAS_rec = np.core.records.fromarrays(zip(*ETAS_array), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	#GMPE_rec = [[x,y,0] for x,y in itertools.product(numpy.linspace(min(lons), max(lons), gmp_size[0]), numpy.linspace(min(lats), max(lats), gmp_size[1]))]
	#
	# so what are the parameters? for now, assume we have a rectangular grid;
	# define parameters from which to construct a GMPE array.
	d_lon = lons[1]-lons[0]
	d_lat = lats[1]-lats[0]		# maybe need something more robust than this, but it should do...
	lon_range = (min(lons), max(lons)+d_lon, d_lon)
	lat_range = (min(lats), max(lats)+d_lat, d_lat)
	l_etas = len(ETAS_rec)
	#
	#d_lat_gmp = (max(lats)-min(lats))/gmp_size[1]
	#d_lon_gmp = (max(lons)-min(lons))/gmp_size[0]
	gmp_lat_range = (min(lats), max(lats), gmp_size[1])
	gmp_lon_range = (min(lons), max(lons), gmp_size[0])
	#
	t0 = time.time()
	#
	# now, we want both SPP and parallel options. SPP should be as much like MPP with one processor as possible, but avoiding the
	# overhead of piping all the data back and forth.
	#
	m_reff=0.
	#
	if n_procs>1:
		#
		# multi-process.
		P = mpp.Pool(n_procs)
		#
		#we want to construct the GMP_{sub}_arrays at the process level, so we don't have to pipe the whole array to the process.
		# note: the [[x,y], ...] part of the array is like:
		# [[x,y] for x,y in itertools.product(np.arange(*lon_range), np.arange(*lat_range))]
		#
		#
		# so we'll need to write calc_gmps() (aka, copy the single process bit).
		chunk_size = int(np.ceil(l_etas/n_procs))		# "chunk" size, or length of sub-arrays for parallel processing.
		#
		# pass part of ETAS; distribute and return a full GMPE_rec[] from each process.
		resultses = [P.apply_async(calc_max_GMPEs, (), {'ETAS_rec':ETAS_rec[j_p*chunk_size:(j_p+1)*chunk_size], 'lon_range':gmp_lon_range, 'lat_range':gmp_lat_range, 'm_reff':5.0, 'just_z':True}) for j_p in range(n_procs)]
		#
		P.close()
		P.join()
		#
		# not sure of this syntax just yet. it is admittedly a little bit convoluted. it might be better to just suck it up and
		# do an extra loop through the array: set up the zero-value initial array, then add all the returns. here, we're trying to
		# squeeze out a little bit of performance by setting up the array and the first results simultaneously.
		#GMPE_rec = np.core.records.fromarrays(zip(*[[x,y,z] for (x,y),z in zip(itertools.product(np.arange(*lon_range), np.arange(*lat_range)), resultses[0].get())]), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
		GMPE_rec = np.core.records.fromarrays(zip(*[[x,y,z] for (x,y),z in zip(itertools.product(np.linspace(*gmp_lon_range), np.linspace(*gmp_lat_range)), resultses[0].get())]), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
		for j,res in enumerate(resultses[1:]):
			GMPE_rec['z']+=np.array(res.get())
			pass
		#
		# look in vc_parser for proper syntax using Pool() objects with close() and join().
		#P.join()
	#
	if n_procs==1:
		# there are different ways to calculate GMPE. just so we can get a number, let's just aggregate the output for now.
		GMPE_rec = calc_max_GMPEs(ETAS_rec=ETAS_rec, lat_range=lat_range, lon_range=lon_range)
	#
	t1 = time.time()
	print(t1 - t0)
	#
	#GMPE_rec.dump(fname_out)
	#
	plot_xyz_image(GMPE_rec, fignum=4, cmap='hot')
	
	return GMPE_rec
#
def calc_max_GMPEs(ETAS_rec=None, lat_range=None, lon_range=None, m_reff=5.0, motion_type="PGA-soil", just_z=False):
	# what is the correct syntax to return a subset of columns of a recarray? (the fastest way, of course)?
	#
	# construct GMP array and calculate GM from ETAS. this function to be used as an mpp.Pool() worker.
	#GMPE_rec =[[x,y,0.] for x,y in itertools.product(np.arange(*lon_range), np.arange(*lat_range))]	# check these for proper
	#
	# create an empty GMPE array.																		# sequenceing and grouping.
	#GMPE_rec = np.core.records.fromarrays(zip(*[[x,y,0.] for x,y in itertools.product(np.arange(*lon_range), np.arange(*lat_range))]), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	GMPE_rec = np.core.records.fromarrays(zip(*[[x,y,0.] for x,y in itertools.product(np.linspace(*lon_range), np.linspace(*lat_range))]), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	#
	d_lat = (lat_range[1]-lat_range[0])/lat_range[2]
	d_lon = (lon_range[1]-lon_range[0])/lon_range[2]
	#
	j_prev=-1
	for (j, (lon1, lat1, z_e)), (k, (lon2, lat2, z_g)) in itertools.product(enumerate(ETAS_rec), enumerate(GMPE_rec)):
		# M=rate_to_m(z_e)
		if j!=j_prev:
			#print('new row[{}]: {}/{}'.format(os.getpid(), j,k))
			j_prev=j
		#
		#M = m_from_rate(z_e, m_reff)
		#M = etas_to_mag(z_e, area=d_lat*d_lon*math.cos(lat1*math.pi/180.)*111.1*111.1, t0=3600., t1=0., t2=30.*24.*3600., p=1.05, dm=1.0, mc=2.5)
		M = 30.+etas_to_mag(10.**(z_e), area=100., t0=3600., t1=0., t2=90.*24.*3600., p=1.05, dm=1.0, mc=2.5)
		#if j<5 and k<5: print('M: ', M)
		#M = 2.0
		#
		distance = spherical_dist(lon_lat_from=[lon1, lat1], lon_lat_to=[lon2, lat2])
		#g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
		#distance = g['s12']
		#	
		#S_Horiz_Soil_Acc = Y(distance, M, "PGA-soil")
		S_Horiz_Soil_Acc = f_Y(distance, M, motion_type)
		#
		#GMPE_rec['z'][k] = max(z_g, S_Horiz_Soil_Acc)
		GMPE_rec['z'][k] += S_Horiz_Soil_Acc
		
		#GMPE_rec['z'][k] = max(z_g*numpy.exp(z_e), abs(S_Horiz_Soil_Acc*numpy.exp(z_e)))		
	#
	#print('finished with GMPE_rec, len={}'.format(len(GMPE_rec)))
	if just_z:
		return GMPE_rec['z']
	else:
		return GMPE_rec
#
#
def plot_xyz_image(xyz, fignum=0, logz=True, interp_type='nearest', cmap='jet'):
	#
	if not hasattr(xyz, 'dtype'):
		xyz = numpy.core.records.fromarrays(zip(*xyz), dtype=[('x','>f8'), ('y','>f8'), ('z','>f8')])
	#
	X = sorted(list(set(xyz['x'])))
	Y = sorted(list(set(xyz['y'])))
	#
	if logz: zz=numpy.log(xyz['z'].copy())
	#zz.shape=(len(Y), len(X))
	zz.shape=(len(X), len(Y))
	#
	plt.figure(fignum)
	plt.clf()
	plt.imshow(zz, interpolation=interp_type, cmap=cmap)
	plt.colorbar()
	
#
def interpolate_etas_test(etas_data='etas_src/etas_japan_20160419_2148CDT_xyz.xyz', new_size=.5, img_interp='nearest', fignum=0):
	# load an etas (or other) data file, plot. interpolate to new lattice, plot again.
	# img_interp: interpretation style for plt.imshow(), basically just for show (not part of the actual interpolation algorithm(s).
	#            options include (‘none’, ‘nearest’, ‘bilinear’, ‘bicubic’, ‘spline16’, ‘spline36’, ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’, ‘quadric’, ‘catrom’, ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’)
	#
	with open(etas_data) as fin:
		xyz = [[float(x) for x in rw.split()] for rw in fin if rw[0] not in (chr(9), chr(13), chr(10), chr(32), '#')]
		#
	#
	print('size: ', len(xyz))
	#
	for j,(x,y,z) in enumerate(xyz): xyz[j][2]=numpy.log(z)
	lons_in, lats_in = (sorted(list(set(col))) for col in list(zip(*xyz))[0:2])
	#Z = numpy.array([numpy.log(z) for x,y,z in xyz])
	Z = numpy.array([z for x,y,z in xyz])
	#
	#lons_in sorted(list(set([x for x,y,z in xyz])))
	#lats_in = sorted(list(set([y for x,y,z in xyz])))
	#
	Z.shape=(len(lats_in), len(lons_in))
	#Z=Z.transpose()
	#
	if fignum!=None:
		plt.figure(fignum)
		plt.clf()
		ax1 = plt.gca()
		ax1.imshow(Z, interpolation=img_interp, cmap='hot')
		plt.title('etas test, input data.')
	#
	# ... and i think this *should* work but it chucks an error. are the data too complex? not sequenced properly? maybe test with
	# the 2D array that we know will plot properly.
	#data_interp = interpolate_RSBS(numpy.array(xyz), sz=(2.*Z.shape[0], 2.*Z.shape[1]), fignum=2)
	#
	# ... but this works quite nicely (using the scipy.interpolate.interp2d() method.
	data_interp = interpolate_scipy(numpy.array(xyz), new_size=new_size, fignum=2)
	#
	return data_interp

def interpolate_scipy(data,new_size=.5, interp_type='cubic', lon1=None, lon2=None, lat1=None, lat2=None, fignum=None, cmap='hot'):
	lons = sorted(list(set([x for x,y,z in data])))
	lats = sorted(list(set([y for x,y,z in data])))
	#
	#print('lls: ', len(lats), len(lons))
	Zs = numpy.reshape([rw[2] for rw in data], (len(lats), len(lons)))
	# we can pass new_size as an array of the new size/shape, or we can pass a number and scale the existing data.
	if isinstance(new_size,int): new_size=float(new_size)
	if isinstance(new_size,float): new_size=[new_size*x for x in Zs.shape]
	#
	#####
	if fignum!=None:
		fg=plt.figure(fignum, figsize=(5,10))
		plt.clf()
		ax1 = fg.add_axes([.1,.08,.8,.4])
		ax2 = fg.add_axes([.1,.5, .8,.4])
		#
		ax1.imshow(Zs, interpolation='nearest', cmap=cmap)
		#
		#plt.title('interpolate(d)_scipy')
		ax1.set_title('original')
		ax2.set_title('interpolate(d)_scipy')
	#####
	#
	lon1 = (lon1 or min(lons))
	lon2 = (lon2 or max(lons))
	lat1 = (lat1 or min(lats))
	lat2 = (lat2 or max(lats))
	#
	new_lats = np.linspace(lat1, lat2, new_size[0])
	new_lons = np.linspace(lon1, lon2, new_size[1])
	#new_lats, new_lons = np.meshgrid(new_lats, new_lons)
	#
	f_int = scipy.interpolate.interp2d(lons, lats, Zs, kind=interp_type)
	Zs_new = f_int(new_lons, new_lats)
	#
	if fignum!=None:
		ax2.imshow(Zs_new, interpolation='nearest', cmap=cmap)
		#
	#
	print('shapes: ', new_lons.shape, new_lats.shape, Zs_new.shape, Zs_new.size)
	
	return numpy.array(list(zip(*(numpy.reshape(X, (X.size,)) for X in (new_lons, new_lats, Zs_new)))))
		

#
def calc_GMPE(lon1, lat1, lon2, lat2, z_etas, m_reff):
	m_reff=0.
	M = m_from_rate(z_e, m_reff)
	#M = 2.0
	#
	distance = spherical_dist(lon_lat_from=[lon1, lat1], lon_lat_to=[lon2, lat2])
	#g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
	#distance = g['s12']
	#	
	S_Horiz_Soil_Acc = f_Y(distance, M, motion_type)
	GMPE_rec['z'][k] = max(GMPE_rec['z'][k], S_Horiz_Soil_Acc)

def m_from_rate(rate=None, m_reff=0.):
	# compute a source magnitude based on ETAS rate and a reference magnitude (or something).
	#
	# but for now, just a constant
	return 2.0
#
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
#
#
def open_xyz_file(fname='etas_src/etas_japan_20160419_2148CDT_xyz.xyz'):
	with open(fname, 'r') as xyz:
		# i don't know why, except maybe because it does handle exceptions/crashes before the file is closed, but this
		# "open()" syntax seems to be recommended over open(), close() implicit blocks.
		#
		ETAS_array = []
		GMPE_array = []
		#
		#ETAS_array = [[float(x), float(y), float(z)] for x,y,z in xyz]
		#ETAS_array = numpy.core.records.fromarrays([[float(x) for x in rw.split()] for rw in xyz if not rw[0] in ('#', chr(32), chr(10), chr(13), chr(9))], dtype=[('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
		return numpy.core.records.fromarrays(zip(*[[float(x) for x in rw.split()] for rw in xyz if not rw[0] in ('#', chr(32), chr(10), chr(13), chr(9))]), dtype=[('x', '>f8'), ('y', '>f8'), ('z', '>f8')])

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




