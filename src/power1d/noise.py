'''
One-dimensional noise models.

Copyright (C) 2023  Todd Pataky
'''




from copy import deepcopy
import inspect
import numpy as np
from . _base import Sample1D, _Continuum1D

# from . primitives import _Primitive,Primitive
# from . signals import _Signal,Signal
from . geom import Continuum1D
from . random import Generator1D
from . _assert import _assert_spm1d



def from_array(noise, x):
	'''
	Create Scaled noise object from a 1D array.
	
	This function is equivalent to creating a Scaled noise object.
	
	While redundant this function is included for API consistency,
	as a sister-function to geom.from_array and
	models.datasample_from_array
	
	Arguments:

	*noise* ---- a Noise object
	
	*x* ---- scaling array (1D numpy array)

	Outputs:
	
	*obj* ---- a Scaled noise object

	Example:

	.. plot::
		:include-source:

		import numpy as np
		import matplotlib.pyplot as plt
		import power1d

		J      = 8  # sample size
		Q      = 101  # number of continuum nodes
		x      = 5.1 + 5 * np.sin( np.linspace(0, 4*np.pi, Q) )
		noise  = power1d.noise.SmoothGaussian( J, Q, fwhm=30 ) # baseline noise model
		snoise = power1d.noise.from_array( noise, x ) # scaled noise object

		fig,axs = plt.subplots(1, 3, figsize=(10,3), tight_layout=True)
		noise.plot( ax=axs[0] )
		axs[1].plot( x )
		snoise.plot( ax=axs[2] )
		labels  = 'Baseline noise model', 'Scaling array', 'Scaled noise'
		[ax.set_title(s) for ax,s in zip(axs,labels)]
		plt.show()
	'''
	assert isinstance(x, np.ndarray), 'x must be a numpy array.'
	assert x.ndim == 1, 'x must be a one-dimensional array.\nAcutal dimensionality: %d' %value.ndim
	assert x.size == noise.Q, f'x must have the same number of elements as the noise object. x has {x.size} elements and noise has {noise.Q} elements.'
	assert np.all( x > 0 ), 'All values in x must be greater than zero.'
	return Scaled(noise, x)


def from_residuals( r, pad=False ):
	'''
	Convenience function for creating Scaled noise objects from sets of
	experimental residuals.
	
	The mean of the residuals must be zero (i.e., the null continuum)
	
	WARNING! As shown in the example below, "from_residuals" may produce a
	noise model that does NOT embody all features of the residuals. In
	this case more complex noise modeling (e.g. Additive, Mixture) may
	be required.
	
	Arguments:

	*r* ---- a (J,Q) array of experimental residuals (J=observations, Q=domain nodes)

	Outputs:
	
	*obj* ---- a Scaled noise object


	Example:

	.. plot::
		:include-source:

		import numpy as np
		import matplotlib.pyplot as plt
		import power1d

		y      = power1d.data.weather()['Atlantic']
		r      = y - y.mean( axis=0 ) # residuals
		snoise = power1d.noise.from_residuals( r ) # scaled noise object

		fig,axs = plt.subplots(1, 3, figsize=(10,3), tight_layout=True)
		axs[0].plot( y.T )
		axs[1].plot( r.T )
		snoise.plot( ax=axs[2] )
		labels  = 'Original data', 'Residuals', 'Scaled noise model'
		[ax.set_title(s) for ax,s in zip(axs,labels)]
		plt.show()
	'''
	assert isinstance(r, np.ndarray), 'r must be a numpy array.'
	assert r.ndim == 2, 'r must be a two-dimensional array.\nAcutal dimensionality: %d' %r.ndim
	_assert_spm1d()
	# create scaled noise model:
	import spm1d
	J,Q   = r.shape
	s     = r.std(axis=0, ddof=1)
	fwhm  = spm1d.geom.estimate_fwhm( r )
	noise = SmoothGaussian(J, Q, fwhm=fwhm, pad=pad)
	return Scaled(noise, s)


class _Noise(Sample1D):
    '''
    Abstract noise class
    '''

    islinked  = False     #whether or not another noise model is linked to this one
    ismaster  = False     #if True, the "other" noise model adopts this noise value
    other     = None      #linked noise model (if any)

    def _get_new_random_value(self):
        self.random()
        return self.value
    def _random(self):        #abstract method (to be implemented by all child classes)
        pass
    def _set_other_value(self):
        self.other.value    = self.value
    def get_iterator(self, iterations=20):
        for i in range(iterations):
            yield self._get_new_random_value()
    def copy(self):
        return deepcopy(self)
    def link(self, other):
        assert isinstance(other, self.__class__), 'Linked and master noise classes must be the same'
        assert self.J == other.J, 'Linked noise class must have same shape as the master noise.'
        assert self.Q == other.Q, 'Linked noise class must have same shape as the master noise.'
        self.islinked       = True
        self.ismaster       = True
        self.other          = other
        self.other.islinked = True
        self._set_other_value()
    def random(self):        #abstract method
        if self.islinked:
            if self.ismaster:
                self._random()
                self._set_other_value()
        else:
            self._random()
    def set_attr(self, attr, value):
        if attr == 'fwhm':
            self._gen.set_fwhm( value )
        else:
            setattr(self, attr, value)
        self.random()


    def set_sample_size(self, J):
        super().set_sample_size(J)
        self.random()
    iterate = get_iterator






class ConstantGaussian(_Noise):
	'''
	Gaussian-distributed constant continuum noise.

	Each of the J continua has a constant value, and these values
	are distributed normally according to *mu* and *sigma*.

	Arguments:

	*J* ---- sample size (int) (default 1)

	*Q* ---- continuum size (int) (default 101)

	*x0* ---- minimum value (float or int) (default 0)

	*x1* ---- maximum value (float or int) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		noise   = power1d.noise.ConstantGaussian( J=8, Q=101, mu=0, sigma=1.0 )
		noise.plot()
	'''
	def __init__(self, J=1, Q=101, mu=0, sigma=1):
		super().__init__(J, Q)
		self._assert_scalar( dict(mu=mu, sigma=sigma) )
		self._assert_greater( dict(sigma=sigma), 0 )
		self.mu    = mu
		self.sigma = sigma
		self.ones  = np.ones((J,Q))
		self._random()

	def _random(self):
		self.value = self.mu + self.sigma * (np.random.randn(self.J)*self.ones.T).T



class ConstantUniform(_Noise):
	'''
	Uniformly-distributed constant continuum noise.

	Each of the J continua has a constant value, and these values
	are distributed uniformly between *x0* and *x1*.

	Arguments:

	*J* ---- sample size (int) (default 1)

	*Q* ---- continuum size (int) (default 101)

	*x0* ---- mean (float or int) (default 0)

	*x1* ---- standard deviation (float or int) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		noise   = power1d.noise.ConstantUniform( J=8, Q=101, x0=-0.5, x1=1.3 )
		noise.plot()
	'''
	def __init__(self, J=1, Q=101, x0=0, x1=1):
		super().__init__(J, Q)
		self._assert_window( dict(x0=x0), dict(x1=x1), -np.inf, +np.inf, asint=False, le=False, ge=False )
		self.x0    = x0
		self.x1    = x1
		# self.dx    = x1 - x0
		self.ones  = np.ones((J,Q))
		self._random()

	def _random(self):
		self.value = self.x0 + self.dx * (np.random.rand(self.J)*self.ones.T).T

	@property
	def dx(self):
		return self.x1 - self.x0




class Gaussian(_Noise):
	'''
	Uncorrelated Gaussian noise.
	
	Arguments:
	
	*J* ---- sample size (int) (default 1)
	
	*Q* ---- continuum size (int) (default 101)
	
	*mu* ---- mean (float or int) (default 0)
	
	*sigma* ---- standard deviation (float or int) (default 1)
	
	
	Example:
	
	.. plot::
		:include-source:
	
		import power1d
	
		noise   = power1d.noise.Gaussian( J=8, Q=101, mu=0, sigma=1.0 )
		noise.plot()
	'''
	def __init__(self, J=1, Q=101, mu=0, sigma=1):
		super().__init__(J, Q)
		self._assert_scalar( dict(mu=mu, sigma=sigma) )
		self._assert_greater( dict(sigma=sigma), 0 )
		self.mu    = mu
		self.sigma = sigma
		self._random()
	def _random(self):
		self.value = self.mu + self.sigma * np.random.randn(self.J, self.Q)






class Skewed(_Noise):
	'''
	Skewed noise.
	
	.. warning:: Skewed distributions are approximate and may not be consistent\
	with theoretical solutions. In particular the their maximum likelihoods are not *mu*.\
	In **power1d** skewed distribution are meant only to be used as tools to approximate\
	experimentally observed skewed noise and / or for exploratory purposes (i.e. to examine\
	power changes associated with roughly skewed distributions).


	Arguments:

	*J* ---- sample size (int) (default 1)

	*Q* ---- continuum size (int) (default 101)

	*mu* ---- mean (float or int) (default 0)

	*sigma* ---- standard deviation (float or int) (default 1)

	*alpha* ---- skewness (float or int) (default 0)


	Modified from a StackOverflow contribution by jamesj629:

	http://stackoverflow.com/questions/36200913/generate-n-random-numbers-from-a-skew-normal-distribution-using-numpy


	Example:

	.. plot::
		:include-source:

		import power1d

		noise   = power1d.noise.Skewed( J=8, Q=101, mu=0, sigma=1.0, alpha=5 )
		noise.plot()
	'''
	def __init__(self, J=8, Q=101, mu=0, sigma=1, alpha=0):
		super().__init__(J, Q)
		self._assert_scalar( dict(mu=mu, sigma=sigma, alpha=alpha) )
		self._assert_greater( dict(sigma=sigma), 0 )
		self.mu    = mu
		self.sigma = sigma
		self.alpha = alpha
		self.skew0 = alpha / (1.0 + alpha**2)**0.5   #first skewness constant
		self.skew1 = (1.0 - self.skew0**2)**0.5      #second skewness constant
		self._random()

	def _random(self):
		u0    = np.random.randn(self.J, self.Q)
		v     = np.random.randn(self.J, self.Q)
		u1    = (self.skew0*u0 + self.skew1*v) * self.sigma
		u1[u0 < 0] *= -1
		self.value  = self.mu + u1




class SmoothGaussian(_Noise):
    '''
    Smooth (correlated) Gaussian noise.


    Arguments:

    *J* ---- sample size (int) (default 1)

    *Q* ---- continuum size (int) (default 101)

    *mu* ---- mean (float or int) (default 0)

    *sigma* ---- standard deviation (float or int) (default 1)

    *fwhm* ---- smoothness (float or int) (default 20); this is the full-width-at-half-maximum of a Gaussian kernel which is convolved with uncorrelated Gaussian noise;  the resulting smooth noise is re-scaled to unit variance

    *pad* ---- whether to pad continuum when smoothing (True or False) (default False);  unpadded noise has the same value at the beginning and end of the continuum



    Example:

    .. plot::
        :include-source:

        import power1d

        noise   = power1d.noise.SmoothGaussian( J=8, Q=101, mu=0, sigma=1.0, fwhm=25, pad=False )
        noise.plot()
    '''
    def __init__(self, J=1, Q=101, mu=0, sigma=1, fwhm=20, pad=False):
        super().__init__(J, Q)
        self._assert_scalar( dict(mu=mu, sigma=sigma, fwhm=fwhm) )
        self._assert_greater( dict(sigma=sigma, fwhm=fwhm), 0 )
        self._assert_bool( dict(pad=pad) )
        self.mu     = mu
        self.sigma  = sigma
        # self.fwhm   = fwhm
        self.pad    = pad
        self._gen   = Generator1D(J, Q, fwhm, pad)
        self._random()
    def _random(self):
        self.value  = self.mu + self.sigma * self._gen.generate_sample()
    @property
    def fwhm(self):
        return self._gen.FWHM
    def set_sample_size(self, J):
        self._gen   = Generator1D(J, self.Q, self.fwhm, self.pad)
        super().set_sample_size(J)






class SmoothSkewed(_Noise):
    '''
    Smooth, skewed noise.

    .. warning:: This smooth skewed distribution implementation is preliminary and will\
    only accept skewness parameter "alpha" values in the range (1, 5). To skew in the\
    opposite direction use (-5, -1).  Note that skewed distributions are approximate\
    and may not be consistent with theoretical solutions.

    Arguments:

    *J* ---- sample size (int) (default 1)

    *Q* ---- continuum size (int) (default 101)

    *mu* ---- mean (float or int) (default 0)

    *sigma* ---- standard deviation (float or int) (default 1)

    *fwhm* ---- smoothness (float or int) (default 20); this is the full-width-at-half-maximum of a Gaussian kernel which is convolved with uncorrelated Gaussian noise;  the resulting smooth noise is re-scaled to unit variance

    *pad* ---- whether to pad continuum when smoothing (True or False) (default False);  unpadded noise has the same value at the beginning and end of the continuum

    *alpha* ---- skewness (float or int between -5 and 5) (default 0)



    Example:

    .. plot::
        :include-source:

        import power1d

        noise   = power1d.noise.SmoothSkewed( J=8, Q=101, mu=0, sigma=1.0, fwhm=25, pad=False, alpha=-3 )
        noise.plot()
    '''
    def __init__(self, J=8, Q=101, mu=0, sigma=1, fwhm=20, pad=True, alpha=0):
        super().__init__(J, Q)
        self._assert_scalar( dict(mu=mu, sigma=sigma, fwhm=fwhm, alpha=alpha) )
        self._assert_bool( dict(pad=pad) )
        self._assert_greater( dict(sigma=sigma), 0 )
        if alpha>0:
            self._assert_bounds( dict(alpha=alpha), 1, 5, ge=True, le=True )
        else:
            self._assert_bounds( dict(alpha=alpha), -5, -1, ge=True, le=True )
        self.mu    = mu
        self.sigma = sigma
        # self.fwhm  = fwhm
        self.pad   = pad
        self.alpha = alpha
        self.skew0 = alpha / (1.0 + alpha**2)**0.5   #first skewness constant
        self.skew1 = (1.0 - self.skew0**2)**0.5      #second skewness constant
        self._gen  = Generator1D(J, Q, fwhm, pad)
        self._random()


    def _get_random_skewed_vector(self):
        u0    = np.random.randn(self.J)
        v     = np.random.randn(self.J)
        u1    = (self.skew0*u0 + self.skew1*v) * self.sigma
        u1[u0 < 0] *= -1
        return self.mu + u1


    def _random(self):
        y     = self.mu + self.sigma * self._gen.generate_sample()
        x     = self._get_random_skewed_vector()
        m     = y.mean(axis=1)
        self.value = ((y.T + self.alpha*x ).T) / self.alpha

    @property
    def fwhm(self):
        return self._gen.FWHM

    def set_sample_size(self, J):
        self._gen   = Generator1D(J, self.Q, self.fwhm, self.pad)
        super().set_sample_size(J)




class Uniform(_Noise):
	'''
	Uniform noise.

	Arguments:

	*J* ---- sample size (int) (default 1)

	*Q* ---- continuum size (int) (default 101)

	*x0* ---- minimum value (float or int) (default 0)

	*x1* ---- maximum value (float or int) (default 1)


	Example:

	.. plot::
		:include-source:

		import power1d

		noise   = power1d.noise.Uniform( J=8, Q=101, x0=0, x1=1.0 )
		noise.plot()
	'''
	def __init__(self, J=1, Q=101, x0=0, x1=1):
		super().__init__(J, Q)
		self._assert_window( dict(x0=x0), dict(x1=x1), -np.inf, +np.inf, asint=False, le=False, ge=False )
		self.x0    = x0
		self.x1    = x1
		# self.dx    = x1 - x0
		self._random()
		
	def _random(self):
		self.value = self.x0 + self.dx * np.random.rand(self.J, self.Q)

	@property
	def dx(self):
		return self.x1 - self.x0


#-----------------------------
#   COMPOUND NOISE CLASSES
#-----------------------------


class _MultiNoise(_Noise):
	def set_sample_size(self, J):
		for m in self.models:
			m.set_sample_size(J)
		super().set_sample_size(J)


class Additive(_MultiNoise):
	'''
	Additive model (sum of two or more other noise types.)

	Arguments:

	A sequence of power1d.noise models (must have the same shape:  (J,Q) )



	Example:

	.. plot::
		:include-source:

		import power1d

		noise0  = power1d.noise.SmoothGaussian( J=8, Q=501, mu=0, sigma=1.0, fwhm=100 )
		noise1  = power1d.noise.Gaussian( J=8, Q=501, mu=0, sigma=0.1 )
		noise   = power1d.noise.Additive(noise0, noise1)
		noise.plot()
	'''
	def __init__(self, *noise_models):
		self._assert_instance_all( dict(noise_models=noise_models), [_Noise] )
		self._assert_same_shape_all( dict(noise_models=noise_models), withJ=True )
		self.models  = list( noise_models )
		super().__init__(self.models[0].J, self.models[0].Q)
		self._random()
	def _random(self):
		y          = np.zeros(  (self.J, self.Q)  )
		for m in self.models:
			m.random()
			y     += m.value
		self.value = y



class Mixture(_MultiNoise):
	'''
	Noise mixture model (mixture of noise types in a fixed ratio)


	Arguments:

	A sequence of power1d.noise models  (must have the same shape:  (,Q) )



	Example:

	.. plot::
		:include-source:

		import power1d

		noise0  = power1d.noise.SmoothGaussian( J=3, Q=101, mu=3, sigma=1.0, fwhm=20 )
		noise1  = power1d.noise.Gaussian( J=5, Q=101, mu=-3, sigma=1.0 )
		noise   = power1d.noise.Mixture(noise0, noise1)
		noise.plot()
	'''
	def __init__(self, *noise_models):
		self._assert_instance_all( dict(noise_models=noise_models), [_Noise] )
		self._assert_same_shape_all( dict(noise_models=noise_models), withJ=False )
		self.models  = list( noise_models )
		J            = sum([m.J  for m in self.models])
		Q            = self.models[0].Q
		super().__init__(J, Q)
		self._random()
	def _random(self):
		y          = []
		for m in self.models:
			m.random()
			y.append( m.value )
		self.value = np.vstack(y)



class Scaled(_Noise):
	'''
	Scaled noise model.

	Arguments:

	*noise* ---- a power1d.noise object

	*scale* ---- a numpy array or a power1d.primitive object


	Example:

	.. plot::
		:include-source:

		import numpy as np
		import power1d

		Q       = 101
		noise0  = power1d.noise.Gaussian( J=5, Q=Q, mu=0, sigma=1.0 )
		scale   = np.linspace(0, 1, Q)
		noise   = power1d.noise.Scaled(noise0, scale)
		noise.plot()
	'''
	def __init__(self, noise, scale):
		self._assert_instance( dict(noise=noise), [_Noise] )
		try:
			scale = Continuum1D(scale)
		except AssertionError:
			pass
		self._assert_instance( dict(scale=scale), [_Continuum1D] )
		self._assert_same_shape( dict(noise=noise), dict(scale=scale), withJ=False )
		self.noise = noise
		self.scale = scale
		super().__init__(noise.J, noise.Q)
		self._random()
	def _random(self):
		self.noise.random()
		self.value  = self.scale.value * self.noise.value
	def set_sample_size(self, J):
		self.noise.set_sample_size(J)
		super().set_sample_size(J)
		



class SignalDependent(_Noise):
	'''
	Signal-dependent noise model.

	Arguments:

	*noise* ---- a power1d.noise object

	*signal* ---- a numpy array or a power1d.signal object

	*fn* ---- an arbitrary function of noise and signal (default fn = lambda n,s: n + (n * s))


	Example:

	.. plot::
		:include-source:

		import numpy as np
		import power1d

		Q       = 101
		noise0  = power1d.noise.Gaussian( J=5, Q=Q, mu=0, sigma=1.0 )
		signal  = power1d.geom.GaussianPulse(Q=Q, q=60, amp=3, fwhm=15)
		fn      = lambda n,s: n + (n * s**3)
		noise   = power1d.noise.SignalDependent(noise0, signal, fn)
		noise.plot()
	'''
	def __init__(self, noise, signal, fn=None):
		self._assert_instance( dict(noise=noise), [_Noise] )
		try:
			signal = Continuum1D(signal)
		except AssertionError:
			pass
		self._assert_instance( dict(signal=signal), [_Continuum1D] )
		self._assert_same_shape( dict(noise=noise), dict(signal=signal), withJ=False )
		### check function definition:
		self._fn_default = fn is None
		if fn is None:
			fn = lambda n,s: n + (n * s)
		self._assert_callable( dict(fn=fn), nargs=2)
		self._assert_function_result( dict(fn=fn), inputs=(noise.value, signal.value), shape=noise.value.shape)
		### construct:
		self.noise  = noise
		self.signal = signal
		self.fn     = fn
		super().__init__(noise.J, noise.Q)
		self._random()
	def _random(self):
		self.noise.random()
		self.value  = self.fn( self.noise.value, self.signal.value )
	def set_sample_size(self, J):
		self.noise.set_sample_size(J)
		super().set_sample_size(J)
		



