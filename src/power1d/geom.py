'''
One-dimensional (1D) continuum geometry primitives.

The classes in this module represent a set of building blocks
for constructing 1D baselines and signals.

More complex 1D geometries can be constructed by combining multiple
primitives through the standard operators: + - * / **
'''

# Copyright (C) 2023  Todd Pataky


from copy import deepcopy
from math import sqrt,pi,log,floor
import numpy as np
from . _base import _Continuum1D




def from_array(x):
	'''
	Create Continuum1D geometry object(s) from a 1D or 2D array.
	
	Arguments:

	*x* ---- 1- or 2-dimensional numpy array
	
	Outputs:
	
	*obj* ---- a Continuum1D object


	Example:

	.. plot::
		:include-source:

		import numpy as np
		import power1d

		x     = np.random.rand( 101 )
		obj   = power1d.geom.from_array( x ) # Continuum1D object
		obj.plot()
	
		x     = np.random.rand( 5, 101 )
		objs  = power1d.geom.from_array( x ) # list of Continuum1D objects
		[obj.plot()  for obj in objs]
	'''
	assert isinstance(x, np.ndarray), 'x must be a numpy array.'
	assert x.ndim in [1,2], 'x must be a one- or two-dimensional array.\nAcutal dimensionality: %d' %value.ndim
	assert x.size > 1, 'x must have more than one element.'
	if x.ndim ==2:
		return [Continuum1D(xx) for xx in x]
	else:
		return Continuum1D(x)



def from_file( fpath ):
	'''
	Create Continuum1D geometry object(s) from a CSV file.
	
	Only CSV files are currently supported.
	
	1D arrays can be saved as either a single row or a
	single column in a CSV file.
	
	2D arrays must be saved with shape (nrow,ncol) where
	nrow is the number of 1D arrays and each row will be
	converted to a Continuum1D object
	
	Arguments:

	*fpath* ---- full path to a CSV file
	
	Outputs:
	
	*obj* ---- a Continuum1D object
	'''
	from . io import file2geom
	return file2geom( fpath )



# Modified from gaussian_filter1d in SciPy scipy/ndimage/filters.py:
# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def gaussian_kernel(sd):
	'''
	Create a Gaussian kernel with the specified standard deviation (sd).
	(This function is modified from **scipy.ndimage.filters.gaussian_filter1d**)


	Arguments:

	*sd* ---- standard deviation of the Gaussian kernel

	Output:

	An array with a length of approximately  ( 4*sd )
	'''
	kw          = int(4.0 * sd + 0.5)   #kernel width
	weights     = np.asarray([0.0] * (2 * kw + 1))
	weights[kw] = 1.0
	sum         = 1.0
	sd          = sd * sd
	# calculate the kernel:
	for ii in range(1, kw + 1):
		tmp     = np.exp(-0.5 * float(ii * ii) / sd)
		weights[kw + ii] = tmp
		weights[kw - ii] = tmp
		sum    += 2.0 * tmp
	for ii in range(2 * kw + 1):
		weights[ii] /= sum
	return weights





#
# #
# # class _GeometryBuilder(_Continuum1D):
# # 	'''
# # 	Abstract primitive class
# # 	'''
# # 	def _check_other(self, other):
# # 		assert isinstance(other, _Continuum1D), 'input argument must be an instance of a class from power1d.primitives'
# # 		assert self.Q==other.Q, 'Continuum lengths must be the same length (base primitive: Q=%d, input primitive: Q=%d).' %(self.Q, other.Q)
# #
# # 	def __add__(self, other):
# # 		new       = self._init_new(other)
# # 		new.value = self.value + other.value
# # 		return new
# # 	def __div__(self, other):
# # 		new       = self._init_new(other)
# # 		x         = other.value
# # 		x[x==0]   = eps
# # 		new.value = self.value / x
# # 		return new
# # 	def __mul__(self, other):
# # 		new       = self._init_new(other)
# # 		new.value = self.value * other.value
# # 		return new
# # 	def __pow__(self, other):
# # 		new       = self._init_new(other)
# # 		new.value = self.value ** other.value
# # 		return new
# # 	def __sub__(self, other):
# # 		new       = self._init_new(other)
# # 		new.value = self.value - other.value
# # 		return new
# #
# # 	def _init_new(self, other):
# # 		self._check_other(other)
# # 		return Primitive( np.zeros(self.Q) )
# #
# # 	def copy(self):
# # 		return deepcopy(self)
# # 	def fliplr(self):
# # 		self.value = self.value[-1::-1]
# # 	def flipud(self, datum=0):
# # 		self.value = -(self.value-datum) + datum
#
#
#


class Continuum1D(_Continuum1D):
	'''
	Manually defined one-dimensional continuum geometry.
	
	Arguments:

	*value* ---- one-dimensional numpy array


	Example:

	.. plot::
		:include-source:

		import numpy as np
		import power1d

		value = np.random.rand( 101 )
		obj   = power1d.geom.Continuum1D( value )
		obj.plot()
	'''
	def __init__(self, value):
		self._assert_array1d( dict(value=value) )
		super().__init__(value.size)
		self.value = value





class Constant(_Continuum1D):
	'''
	Constant value continuum with amplitude *amp*.

	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*amp* ---- continuum value (float or int) (default 0)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.Constant(Q=101, amp=1.3)
		obj.plot()
	'''
	def __init__(self, Q=101, amp=0):
		self._assert_scalar(  dict(amp=amp)  )
		self.amp    = amp
		super().__init__(Q)

	def _build(self):
		self.value      = self.amp * np.ones(self.Q)






class Exponential(_Continuum1D):
	'''
	Exponentially increasing (or decreasing) continuum

	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*x0* ---- value at starting continuum point (float or int) (default 0)

	*x1* ---- value at ending continuum point (float or int) (default 1)

	*rate* ---- rate of exponential increase (float or int) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.Exponential(Q=101, x0=0.3, x1=3.6, rate=2.8)
		obj.plot()
	'''
	def __init__(self, Q=101, x0=0, x1=1, rate=1):
		self._assert_scalar(  dict(x0=x0, x1=x1, rate=rate)  )
		self._assert_greater( dict(rate=rate), 0  )
		self.x0     = float(x0)
		self.x1     = float(x1)
		self.rate   = float(rate)
		super().__init__(Q)

	def _build(self):
		y               = np.exp( np.linspace(0, self.rate, self.Q) )
		y               = (y - 1) / (y[-1]-1)
		self.value      = self.x0 + (y * (self.x1-self.x0))





class ExponentialSaw(Exponential):
	'''
	Exponentially increasing (or decreasing) continuum with a saw cutoff

	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*x0* ---- value at starting continuum point (float or int) (default 0)

	*x1* ---- value at ending continuum point (float or int) (default 1)

	*rate* ---- rate of exponential increase (float or int) (default 1)

	*cutoff* ---- continuum point at which the continuum returns to *x0* (int) (default 70)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.ExponentialSaw(Q=101, x0=0.3, x1=3.6, rate=2.8, cutoff=85)
		obj.plot()
	'''
	def __init__(self, Q=101, x0=0, x1=1, rate=1, cutoff=70):
		self._assert_Q(Q)
		self._assert_integer(  dict(cutoff=cutoff)  )
		self._assert_bounds(  dict(cutoff=cutoff), 0, Q  )
		self.cutoff = cutoff
		super().__init__(Q, x0, x1, rate)

	def _build(self):
		super()._build()
		self.value[self.cutoff:] = self.x0




class GaussianPulse(_Continuum1D):
	'''
	A Gaussian pulse at a particular continuum location.

	NOTE: one of *fwhm* and *sigma* must be specified, and the other must be None.


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q* ---- pulse center (int) (default 50)

	*fwhm* ---- pulse's full-width-at-half-maximum (float or int) (default None)

	*sigma* ---- pulse's standard deviation (float or int) (default None)

	*amp* ---- pulse amplitude


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.GaussianPulse( Q=101, q=35, fwhm=10, amp=5 )
		obj.plot()
	'''

	def __init__(self, Q=101, q=50, fwhm=None, sigma=None, amp=1):
		if (sigma is None) and (fwhm is None):
			sigma = 5
		### field size parameters:
		self._assert_Q(Q)
		self._assert_integer(  dict(q=q)  )
		self._assert_bounds(  dict(q=q), -Q, 2*Q  )
		### kernel parameters:
		self._assert_one_of_two_none( dict(sigma=sigma, fwhm=fwhm) )
		if sigma is None:
			self._assert_scalar(  dict(fwhm=fwhm)  )
		else:
			self._assert_scalar(  dict(sigma=sigma)  )
		self._assert_scalar(  dict(amp=amp)  )
		### set attributes
		self.q     = q
		self.fwhm  = None if fwhm is None else float(fwhm)
		self.sigma = None if sigma is None else float(sigma)
		self.amp   = float(amp)
		super().__init__(Q)


	def _build(self):
		### initialize signal:
		y               = np.zeros(self.Q)
		sigma           = self.fwhm / (2 * (2*log(2))**0.5 )  if (self.sigma is None) else self.sigma
		signal          = gaussian_kernel(sigma)
		signal_max      = signal.max()
		### move the signal to the appropriate continuum location:
		n               = len(signal)
		q0              = self.q - int(n/2)
		q1              = q0 + n
		if q0 < 0:
			signal      = signal[-q0:]
			q0          = 0
		if q1 > self.Q:
			nextra      = self.Q - q1
			signal      = signal[:nextra]
			q1          = self.Q
		q0,q1           = int(q0), int(q1)
		y[q0:q1]        = signal
		self.value      = y * self.amp / signal_max




class Linear(_Continuum1D):
	'''
	Linearly increasing (or decreasing) continuum

	NOTE: one of *x1* and *slope* must be specified, and the other must be None.


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*x0* ---- value at starting continuum point (float or int) (default 0)

	*x1* ---- value at ending continuum point (float or int) (default None)

	*slope* ---- rate of linear increase (float or int) (default None)

	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.Linear( Q=101, x0=1.2, x1=-4.5 )
		obj.plot()
	'''
	def __init__(self, Q=101, x0=0, x1=None, slope=None):
		if (x1 is None) and (slope is None):
			x1 = 1
		self._assert_scalar(  dict(x0=x0)  )
		self._assert_one_of_two_none( dict(x1=x1, slope=slope) )
		if x1 is None:
			self._assert_scalar(  dict(slope=slope)  )
		else:
			self._assert_scalar(  dict(x1=x1)  )
		self.x0         = x0
		self.x1         = x1
		self.slope      = slope
		super().__init__(Q)

	def _build(self):
		x1              = self.x0 + self.slope*self.Q if self.x1 is None else self.x1
		self.value      = np.linspace(self.x0, x1, self.Q)






class Null(_Continuum1D):
	'''
	Null continuum.  This is equivalent to a constant continuum with amp=0
	
	Keyword arguments:
	
	*Q* ---- continuum size (int) (default: 101)

	Example:
	
	.. plot::
		:include-source:
	
		import power1d
	
		obj = power1d.geom.Null( Q=101 )
		obj.plot()
	'''
	pass




class SawPulse(_Continuum1D):
	'''
	Linearly increasing (or decreasing) continuum, cut-off at a particular point


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q0* ---- continuum point at which the saw starts (int) (default 50)

	*q1* ---- continuum point at which the saw end (int) (default 75)

	*x0* ---- value at start point (float or int) (default 0)

	*x1* ---- value at end point (float or int) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.SawPulse( Q=101, q0=30, q1=80, x0=-1.2, x1=4.5 )
		obj.plot()
	'''
	def __init__(self, Q=101, q0=50, q1=75, x0=0, x1=1):
		self._assert_Q(Q)
		self._assert_window( dict(q0=q0), dict(q1=q1), 0, Q, asint=True, ge=True, le=True )
		self._assert_scalar(  dict(x0=x0, x1=x1)  )
		self.q0         = q0
		self.q1         = q1
		self.x0         = x0
		self.x1         = x1
		super().__init__(Q)

	def _build(self):
		q0,q1           = self.q0, self.q1
		y               = self.x0 * np.ones(self.Q)
		y[q0:q1]        = np.linspace(self.x0, self.x1, q1-q0)
		self.value      = y



class SawTooth(_Continuum1D):
	'''
	Repeating saw pattern.


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q0* ---- continuum point at which the saw starts (int) (default 50)

	*q1* ---- continuum point at which the saw ends (int) (default 75)

	*x0* ---- value at start point (float or int) (default 0)

	*x1* ---- value at end point (float or int) (default 1)

	*dq* ---- inter-tooth distance (int) (default 0)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.SawTooth( Q=101, q0=20, q1=35, x0=0.1, x1=0.9, dq=5 )
		obj.plot()
	'''
	def __init__(self, Q=101, q0=50, q1=75, x0=0, x1=1, dq=0):
		self._assert_Q(Q)
		self._assert_window( dict(q0=q0), dict(q1=q1), 0, Q, asint=True, ge=True, le=True )
		self._assert_bounds(  dict(dq=dq), -1, Q )
		self._assert_scalar(  dict(x0=x0, x1=x1)  )
		self.q0         = q0
		self.q1         = q1
		self.x0         = x0
		self.x1         = x1
		self.dq         = dq
		super().__init__(Q)

	def _build(self):
		q0,q1,dq        = self.q0, self.q1, self.dq
		y               = self.x0 * np.ones(2*self.Q)
		i,w             = q0, q1-q0
		while i < self.Q:
			y[i:i+w]    = np.linspace(self.x0, self.x1, w)
			i          += w + dq
		self.value      = y[:self.Q]



class Sigmoid(_Continuum1D):
	'''
	Sigmoidal step pulse


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q0* ---- continuum point at which the step starts (int) (default 50)

	*q1* ---- continuum point at which the step ends (int) (default 75)

	*x0* ---- value at start point (float or int) (default 0)

	*x1* ---- value at end point (float or int) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.Sigmoid( Q=101, q0=20, q1=55, x0=0.1, x1=0.9 )
		obj.plot()
	'''
	def __init__(self, Q=101, q0=50, q1=75, x0=0, x1=1):
		self._assert_Q(Q)
		self._assert_window( dict(q0=q0), dict(q1=q1), 0, Q, asint=True, ge=True, le=True )
		self._assert_scalar(  dict(x0=x0, x1=x1)  )
		self.q0         = q0
		self.q1         = q1
		self.x0         = x0
		self.x1         = x1
		super().__init__(Q)

	def _build(self):
		q0,q1      = self.q0, self.q1
		z          = np.zeros(self.Q)
		zz         = 6
		z[:q0]     = -zz
		z[q1:]     = +zz
		z[q0:q1]   = np.linspace(-zz, zz, q1-q0)
		y          = 1.0 / (1.0 + np.exp(-1.0 * z))
		y          = (y - y[0]) / (y[-1]-y[0])
		y          = self.x0 + (y * (self.x1-self.x0))
		self.value = y




class Sinusoid(_Continuum1D):
	'''
	Repeating sinusoidal wave


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q0* ---- continuum point at which phase is zero (int) (default 0)

	*amp* ---- amplitude (float or int) (default 1)

	*hz* ---- frequency relative to continuum size (float or int) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.Sinusoid( Q=101, q0=0, amp=1.3, hz=2 )
		obj.plot()
	'''
	def __init__(self, Q=101, q0=0, amp=1, hz=1):
		self._assert_Q(Q)
		self._assert_integer(  dict(q0=q0)  )
		self._assert_bounds(  dict(q0=q0), -1, Q )
		self._assert_scalar(  dict(amp=amp, hz=hz)  )
		self._assert_greater(  dict(amp=amp, hz=hz), 0  )
		self.q0         = q0
		self.amp        = amp
		self.hz         = float(hz)
		super().__init__(Q)

	def _build(self):
		t0         = 2*pi* (1 - self.q0 * self.hz / self.Q)
		t1         = t0 + 2*pi*self.hz
		q          = np.linspace(t0, t1, self.Q)
		self.value =  self.amp * np.sin(q)






class SquarePulse(_Continuum1D):
	'''
	Square pulse


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q0* ---- continuum point at which pulse starts (int) (default 50)

	*q1* ---- continuum point at which pulse ends (int) (default 75)

	*x0* ---- starting value (int or float) (default 0)

	*x1* ---- pulse height (int or float) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.SquarePulse( Q=101, q0=50, q1=70, x0=1.3, x1=2.5 )
		obj.plot()
	'''
	def __init__(self, Q=101, q0=50, q1=75, x0=0, x1=1):
		self._assert_Q(Q)
		self._assert_window( dict(q0=q0), dict(q1=q1), 0, Q, asint=True, ge=True, le=True )
		self._assert_scalar(  dict(x0=x0, x1=x1)  )
		self.q0         = q0
		self.q1         = q1
		self.x0         = x0
		self.x1         = x1
		super().__init__(Q)

	def _build(self):
		q0,q1           = self.q0, self.q1
		y               = self.x0 * np.ones(self.Q)
		y[q0:q1]        = self.x1
		self.value      = y



class SquareTooth(_Continuum1D):
	'''
	Repeating square pulses


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q0* ---- continuum point at which pulse starts (int) (default 10)

	*q1* ---- continuum point at which pulse ends (int) (default 25)

	*x0* ---- starting value (int or float) (default 0)

	*x1* ---- pulse height (int or float) (default 1)

	*dq* ---- inter-tooth distance (int) (default 5)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.SquareTooth( Q=101, q0=30, q1=40, x0=1.3, x1=2.5, dq=15 )
		obj.plot()
	'''
	def __init__(self, Q=101, q0=10, q1=25, x0=0, x1=1, dq=5):
		self._assert_Q(Q)
		self._assert_window( dict(q0=q0), dict(q1=q1), 0, Q, asint=True, ge=True, le=True )
		self._assert_scalar(  dict(x0=x0, x1=x1)  )
		self.q0         = q0
		self.q1         = q1
		self.x0         = x0
		self.x1         = x1
		self.dq         = dq
		super().__init__(Q)

	def _build(self):
		q0,q1           = self.q0, self.q1
		y               = self.x0 * np.ones(2*self.Q)
		i,w             = q0, q1-q0
		while i < self.Q:
			i1          = i + w
			y[i:i1]     = self.x1
			i          += w+self.dq
		self.value      = y[:self.Q]


class TrianglePulse(_Continuum1D):
	'''
	Triangular pulse


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q0* ---- continuum point at which pulse starts (int) (default 50)

	*q1* ---- continuum point at which pulse ends (int) (default 75)

	*x0* ---- starting value (int or float) (default 0)

	*x1* ---- pulse height (int or float) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.TrianglePulse( Q=101, q0=50, q1=70, x0=1.3, x1=2.5 )
		obj.plot()
	'''
	def __init__(self, Q=101, q0=50, q1=75, x0=0, x1=1):
		self._assert_Q(Q)
		self._assert_window( dict(q0=q0), dict(q1=q1), 0, Q, asint=True, ge=True, le=True )
		self._assert_scalar(  dict(x0=x0, x1=x1)  )
		self.q0         = q0
		self.q1         = q1
		self.x0         = x0
		self.x1         = x1
		super().__init__(Q)

	def _build(self):
		q0,q1           = self.q0, self.q1
		y               = self.x0 * np.ones(self.Q)
		dq0             = int(round( 0.5*(q1-q0) ))
		dq1             = q1 - q0 - dq0
		qq              = q0 + dq0
		y[q0:qq+1]      = np.linspace(self.x0, self.x1, dq0+1)
		y[qq:q1]        = np.linspace(self.x1, self.x0, dq1)
		self.value      = y


class TriangleTooth(_Continuum1D):
	'''
	Repeating triangular pulses


	Keyword arguments:

	*Q* ---- continuum size (int) (default: 101)

	*q0* ---- continuum point at which pulse starts (int) (default 10)

	*q1* ---- continuum point at which pulse ends (int) (default 25)

	*x0* ---- starting value (int or float) (default 0)

	*x1* ---- pulse height (int or float) (default 1)

	*dq* ---- inter-tooth distance (int) (default 5)


	Example:

	.. plot::
		:include-source:

		import power1d

		obj = power1d.geom.TriangleTooth( Q=101, q0=30, q1=40, x0=1.3, x1=2.5, dq=15 )
		obj.plot()
	'''
	def __init__(self, Q=101, q0=50, q1=75, x0=0, x1=1, dq=20):
		self._assert_Q(Q)
		self._assert_window( dict(q0=q0), dict(q1=q1), 0, Q, asint=True, ge=True, le=True )
		self._assert_bounds(  dict(dq=dq), -1, Q )
		self._assert_scalar(  dict(x0=x0, x1=x1)  )
		self.q0         = q0
		self.q1         = q1
		self.x0         = x0
		self.x1         = x1
		self.dq         = dq
		super().__init__(Q)

	def _build(self):
		q0,q1           = self.q0, self.q1
		y               = self.x0 * np.ones(2*self.Q)
		i,w             = q0, q1-q0
		while i < self.Q:
			i1          = i + int( w/2 )
			i2          = i1 + int( w/2 )
			n1          = int(w / 2) + 1
			n2          = int(w / 2)
			y[i:i1+1]   = np.linspace(self.x0, self.x1, n1)
			y[i1:i2]    = np.linspace(self.x1, self.x0, n2)
			i           = i2 + self.dq - 1
		self.value      = y[:self.Q]


