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
#
def etas_to_GM(etas_src='../globalETAS/etas_outputs/etas_xyz.xyz', fname_out='GMPE_rec.p', motion_type='PGA-soil', etas_size=None, gmp_size=None, n_procs=None):
	# "ETAS to Ground-Motion:
	# etas_size: if None, use raw data as they are. otherwise, re-size the lattice using scipy interpolation tools (grid_data() i think)
	# gmp_size: if None, use raw (etas) data size, otherwise, create a grid... and for these two variables, we need to decide if we want
	# to define the grid-size, or n_x/n_y, or have options for either. probably, handle a single number as a grid-size (and assume square);
	# handle an array as lattice dimensions.
	#
	#xyz = open('../globalETAS/etas_outputs/etas_xyz.xyz', 'r')
	n_procs=(n_procs or mpp.cpu_count())	
	#
	#
	print('open etas_src file: ', etas_src)
	#
	with open(etas_src, 'r') as xyz:
		# i don't know why, except maybe because it does handle exceptions/crashes before the file is closed, but this
		# "open()" syntax seems to be recommended over open(), close() implicit blocks.
		#
		ETAS_array = []
		GMPE_array = []
		#
		#ETAS_array = [[float(x), float(y), float(z)] for x,y,z in xyz]
		ETAS_array = [[float(x) for x in rw.split()] for rw in xyz if not rw[0] in ('#', chr(32), chr(10), chr(13), chr(9))]
		#
	lons = sorted(list(set([x for x,y,z in ETAS_array])))
	lats = sorted(list(set([y for x,y,z in ETAS_array])))
	#
	######################################
	# resize/interpolate?::
	# 
	# are we re-sizing the lattice? for most cases, we can probably use a cartesian approximation, but scipy happens to have a
	# cartesian-spherical (aka, lat/lon) interpolator:
	# scipy.interpolate.RectSphereBivariateSpline
	#
	if isinstance(etas_size,int) or isinstance(etas_size,float): etas_size=(etas_size, etas_size)
	if isinstance(gmp_size,int) or isinstance(gmp_size,float): gmp_size=(gmp_size, gmp_size)
	#
	if etas_size == None: etas_size=ETAS_array.shape()
	if gmp_size  == None: gmp_size=etas_size
	#
	if not tuple(etas_size)==tuple(ETAS_array.shape()) and False:
		# we're resizing and interpolating.
		ETAS_array = resize_interpolate(ETAS_array, etas_size)
		#
	#
	##############
	#
	GMPE_array = [[x,y,0.] for x,y,z in ETAS_array]
	#
		#xyz.close()
	#
	print('etas_src loaded. load data into arrays and process.')
	#
	ETAS_rec = np.core.records.fromarrays(zip(*ETAS_array), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	#
	# so what are the parameters? for now, assume we have a rectangular grid;
	# define parameters from which to construct a GMPE array.
	d_lon = lons[1]-lons[0]
	d_lat = lats[1]-lats[0]		# maybe need something more robust than this, but it should do...
	lon_range = (min(lons), max(lons)+d_lon, d_lon)
	lat_range = (min(lats), max(lats)+d_lat, d_lat)
	l_etas = len(ETAS_rec)
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
		
		resultses = [P.apply_async(calc_GMPEs, (), {'ETAS_rec':ETAS_rec[j_p*chunk_size:(j_p+1)*chunk_size], 'lon_range':lon_range, 'lat_range':lat_range, 'm_reff':5.0, 'just_z':True}) for j_p in range(n_procs)]
		
		P.close()
		#
		# not sure of this syntax just yet. it is admittedly a little bit convoluted. it might be better to just suck it up and
		# do an extra loop through the array: set up the zero-value initial array, then add all the returns. here, we're trying to
		# squeeze out a little bit of performance by setting up the array and the first results simultaneously.
		GMPE_rec = np.core.records.fromarrays(zip(*[[x,y,z] for (x,y),z in zip(itertools.product(np.arange(*lon_range), np.arange(*lat_range)), resultses[0].get())]), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
		for j,res in enumerate(resultses[1:]):
			GMPE_rec['z']+=np.array(res.get()['z'])
			pass
		#
		# look in vc_parser for proper syntax using Pool() objects with close() and join().
		#P.join()
	#
	if n_procs==1:
		GMPE_rec = calc_GMPEs(ETAS_rec=ETAS_rec, lat_range=None, lon_range=None)
	#
	t1 = time.time()
	print(t1 - t0)
	#
	GMPE_rec.dump(fname_out)
#
def resize_interpolate(ary_in, new_size):
	if hasattr(ary_in, 'dtype'):
		lons = sorted(list(set(ary_in['x'])))
		lats = sorted(list(set(ary_in['y'])))
		zs   = ary_in['z']
	else:
		lons, lats, zs= (numpy.array(sorted(list(set(x)))) for x in (zip(*ary_in)))
		#print('lls: ', lons, lats)
		zs = numpy.array([rw[2] for rw in ary_in])
	#	 
	new_lons = numpy.linspace(min(lons), max(lons), new_size[0])*numpy.pi/180.
	new_lats = numpy.linspace(min(lats), max(lats), new_size[1])*numpy.pi/180.
	#return new_lons, new_lats
	new_lats, new_lons = numpy.meshgrid(new_lats, new_lons)
	#
	data = numpy.array(zs)
	data.shape=(numpy.size(lats), numpy.size(lons))		# or is it len(lats), len(lons) (yes, i think it is)
	lut = RectSphereBivariateSpline(lats, lons, data)
	#data_interp = lut.ev(new_lats.ravel(), new_lons.ravel())
	data_interp = lut.ev(new_lats.ravel(), new_lons.ravel()).reshape(new_size).T
	data_interp = data_interp.reshape((data_interp.size,))
	#return data_interp
	#
	#
	return np.core.records.fromarrays(zip(*[[x*180/numpy.pi,y*180/numpy.pi,z] for (x,y),z in zip(itertools.product(new_lons.reshape((new_lons.size,)), new_lats.reshape((new_lats.size,))), data_interp)]), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
#
def interpolation_test():
	# a short unit-type test of the resize_interpolate() bit.
	# borrowed from :
	# http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.html#scipy.interpolate.RectSphereBivariateSpline
	lats = np.linspace(10, 170, 9) * np.pi / 180.
	lons = np.linspace(0, 350, 18) * np.pi / 180.
	data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T, np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T
	#sh=data.shape
	print('data shape: ', data.shape, data.size)
	#return data
	#data.shape=(data.size)
	#
	dtas = [[x,y,z] for (x,y),z in zip(itertools.product(lons, lats),numpy.reshape(data, numpy.size(data)))]
	print('shape: ', numpy.shape(dtas))
	#
	new_sh = (360, 180)
	new_data = resize_interpolate(dtas, new_sh)
	#print('nds: ', numpy.size(new_data.size))
	#
	img_data = new_data['z'].reshape(new_sh)
	#
	fig = plt.figure(0)
	plt.clf()
	ax1 = fig.add_subplot(211)
	ax1.imshow(data, interpolation='nearest')
	ax2 = fig.add_subplot(212)
	ax2.imshow(img_data, interpolation='nearest')
	plt.show()
	#
	pass
def interpolation_test_2(fignum=0):
	# this needs some tuning, but it appears to be working and semi-functional
	# Suppose we have global data on a coarse grid
	import numpy as np

	lats = np.linspace(10, 170, 9) * np.pi / 180.
	lons = np.linspace(0, 350, 18) * np.pi / 180.
	data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
		          np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T
	print('orig. shape:: ', numpy.shape(data))
	# We want to interpolate it to a global one-degree grid
	dtas = [[x,y,z] for (x,y),z in zip(itertools.product(lons, lats),numpy.reshape(data, numpy.size(data)))]
	
	#data2 = numpy.reshape([rw[2] for rw in dtas], data.shape)
	
	#lns = sorted(list(set([x for x,y,z in dtas])))
	#lts = sorted(list(set([y for x,y,z in dtas])))
	#
	return interpolate2(dtas, (180,360), 1., 360., 1., 180., fignum=fignum)

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
		ax1.imshow(Z, interpolation=img_interp)
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

def interpolate_scipy(data,new_size=.5, interp_type='cubic', lon1=None, lon2=None, lat1=None, lat2=None, fignum=None):
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
		fg=plt.figure(fignum, fig_size=(5,10))
		plt.clf()
		ax1 = fg.add_axes([.1,.08,.8,.4])
		ax2 = fg.add_axes([.1,.5, .8,.4])
		#
		ax1.imshow(Zs, interpolation='nearest')
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
		ax2.imshow(Zs_new, interpolation='nearest')
		#
	#
	print('shapes: ', new_lons.shape, new_lats.shape, Zs_new.shape, Zs_new.size)
	
	return numpy.array(list(zip((numpy.reshape(X, (X.size,)) for X in (new_lons, new_lats, Zs_new)))))
	
	
def interpolate_RSBS(data,sz, lon1=None, lon2=None, lat1=None, lat2=None, fignum=None):
	# interpolate using scipy.interpolate.RectSphereBivariateSpline
	#
	# this needs some tuning, but it appears to be working and semi-functional
	# ... but tends to break for complex array. let's try scipy.interp2d():
	#http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.interp2d.html
	#
	print('data: ', data[0:5])
	#
	lons = sorted(list(set([x for x,y,z in data])))
	lats = sorted(list(set([y for x,y,z in data])))
	#print('lls: ', len(lats), len(lons))
	data = numpy.reshape([rw[2] for rw in data], (len(lats), len(lons)))
	#print('sh: ', numpy.shape(data))
	#####
	plt.figure(fignum+2)
	plt.clf()
	ax1 = plt.gca()
	ax1.imshow(data, interpolation='nearest')
	#####
	#
	lon1 = (lon1 or min(lons))
	lon2 = (lon2 or max(lons))
	lat1 = (lat1 or min(lats))
	lat2 = (lat2 or max(lats))
	#
	#new_lats = np.linspace(1, 180, 180) * np.pi / 180
	#new_lons = np.linspace(1, 360, 360) * np.pi / 180
	new_lats = np.linspace(lat1, lat2, sz[0]) * np.pi / 180
	new_lons = np.linspace(lon1, lon2, sz[1]) * np.pi / 180
	new_lats, new_lons = np.meshgrid(new_lats, new_lons)

	# We need to set up the interpolator object

	#from scipy.interpolate import RectSphereBivariateSpline
	lut = RectSphereBivariateSpline(lats, lons, data)
	#
	#lut = RectBivariateSpline(lats, lons, data)
	print(len(new_lats), len(new_lons[0]), len(data), len(data[0]), new_lats.shape)

	# Finally we interpolate the data.  The `RectSphereBivariateSpline` object
	# only takes 1-D arrays as input, therefore we need to do some reshaping.

	data_interp = lut.ev(new_lats.ravel(), new_lons.ravel()).reshape(new_lats.shape).T

	# Looking at the original and the interpolated data, one can see that the
	# interpolant reproduces the original data very well:
	#
	if fignum!=None:
		fig = plt.figure(fignum+0)
		ax1 = fig.add_subplot(211)
		ax1.imshow(data, interpolation='nearest')
		ax2 = fig.add_subplot(212)
		ax2.imshow(data_interp, interpolation='nearest')
		plt.show()
		#
		# Chosing the optimal value of ``s`` can be a delicate task. Recommended
		# values for ``s`` depend on the accuracy of the data values.  If the user
		# has an idea of the statistical errors on the data, she can also find a
		# proper estimate for ``s``. By assuming that, if she specifies the
		# right ``s``, the interpolator will use a spline ``f(u,v)`` which exactly
		# reproduces the function underlying the data, she can evaluate
		# ``sum((r(i,j)-s(u(i),v(j)))**2)`` to find a good estimate for this ``s``.
		# For example, if she knows that the statistical errors on her
		# ``r(i,j)``-values are not greater than 0.1, she may expect that a good
		# ``s`` should have a value not larger than ``u.size * v.size * (0.1)**2``.

		# If nothing is known about the statistical error in ``r(i,j)``, ``s`` must
		# be determined by trial and error.  The best is then to start with a very
		# large value of ``s`` (to determine the least-squares polynomial and the
		# corresponding upper bound ``fp0`` for ``s``) and then to progressively
		# decrease the value of ``s`` (say by a factor 10 in the beginning, i.e.
		# ``s = fp0 / 10, fp0 / 100, ...``  and more carefully as the approximation
		# shows more detail) to obtain closer fits.

		# The interpolation results for different values of ``s`` give some insight
		# into this process:

		fig2 = plt.figure(fignum+1)
		s = [3e9, 2e9, 1e9, 1e8]
		for ii in range(len(s)):
			#lut = RectSphereBivariateSpline(lats, lons, data, s=s[ii])
			RectBivariateSpline(lats, lons, data, s=s[ii])
			data_interp = lut.ev(new_lats.ravel(),
				                 new_lons.ravel()).reshape(new_lons.shape).T
			ax = fig2.add_subplot(2, 2, ii+1)
			ax.imshow(data_interp, interpolation='nearest')
			ax.set_title("s = %g" % s[ii])
	plt.show()
#
def calc_GMPEs(ETAS_rec=None, lat_range=None, lon_range=None, m_reff=5.0, motion_type="PGA-soil", just_z=False):
	# what is the correct syntax to return a subset of columns of a recarray? (the fastest way, of course)?
	#
	# construct GMP array and calculate GM from ETAS. this function to be used as an mpp.Pool() worker.
	#GMPE_rec =[[x,y,0.] for x,y in itertools.product(np.arange(*lon_range), np.arange(*lat_range))]	# check these for proper
	#
	# create an empty GMPE array.																		# sequenceing and grouping.
	GMPE_rec = np.core.records.fromarrays(zip(*[[x,y,0.] for x,y in itertools.product(np.arange(*lon_range), np.arange(*lat_range))]), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	#GMPE_rec = np.core.records.fromarrays(zip(*GMPE_array), dtype = [('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
	#
	j_prev=0
	for (j, (lon1, lat1, z_e)), (k, (lon2, lat2, z_g)) in itertools.product(enumerate(ETAS_rec), enumerate(GMPE_rec)):
		# M=rate_to_m(z_e)
		if j!=j_prev:
			print('new row[{}]: {}'.format(os.getpid(), j))
			j_prev=j
		#
		M = m_from_rate(z_e, m_reff)
		#M = 2.0
		#
		distance = spherical_dist(lon_lat_from=[lon1, lat1], lon_lat_to=[lon2, lat2])
		#g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
		#distance = g['s12']
		#	
		#S_Horiz_Soil_Acc = Y(distance, M, "PGA-soil")
		S_Horiz_Soil_Acc = f_Y(distance, M, motion_type)
		GMPE_rec['z'][k] = max(GMPE_rec['z'][k], S_Horiz_Soil_Acc)
	#
	if just_z:
		return GMPE_rec['z']
	else:
		return GMPE_rec

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




