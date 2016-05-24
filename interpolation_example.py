'''
# some development, example, and experimental scrpts for 2D interpolation.
#
'''

def main():
	# Suppose we have global data on a coarse grid
	import numpy as np

	lats = np.linspace(10, 170, 9) * np.pi / 180.
	lons = np.linspace(0, 350, 18) * np.pi / 180.
	data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
		          np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

	# We want to interpolate it to a global one-degree grid

	new_lats = np.linspace(1, 180, 180) * np.pi / 180
	new_lons = np.linspace(1, 360, 360) * np.pi / 180
	new_lats, new_lons = np.meshgrid(new_lats, new_lons)

	# We need to set up the interpolator object

	from scipy.interpolate import RectSphereBivariateSpline
	lut = RectSphereBivariateSpline(lats, lons, data)

	# Finally we interpolate the data.  The `RectSphereBivariateSpline` object
	# only takes 1-D arrays as input, therefore we need to do some reshaping.

	data_interp = lut.ev(new_lats.ravel(),
		                 new_lons.ravel()).reshape((360, 180)).T

	# Looking at the original and the interpolated data, one can see that the
	# interpolant reproduces the original data very well:

	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.imshow(data, interpolation='nearest')
	ax2 = fig.add_subplot(212)
	ax2.imshow(data_interp, interpolation='nearest')
	plt.show()

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

	fig2 = plt.figure()
	s = [3e9, 2e9, 1e9, 1e8]
	for ii in range(len(s)):
		lut = RectSphereBivariateSpline(lats, lons, data, s=s[ii])
		data_interp = lut.ev(new_lats.ravel(),
		                     new_lons.ravel()).reshape((360, 180)).T
		ax = fig2.add_subplot(2, 2, ii+1)
		ax.imshow(data_interp, interpolation='nearest')
		ax.set_title("s = %g" % s[ii])
	plt.show()

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
#
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
if __name__=='__main__':
	return main()
else:
	plt.ion()
