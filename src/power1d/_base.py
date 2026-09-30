'''
Base classes for all **power1d** objects.

Copyright (C) 2023  Todd Pataky
'''





import inspect
import numpy as np

eps  = np.finfo(float).eps




class _Power1DObject(object):
	def _to_dict(self):
		from . _serial import to_dict
		return to_dict(self)
	@staticmethod
	def _from_dict(d):
		from . _serial import from_dict
		return from_dict(d)



class _ContniuumObject(_Power1DObject):
	
	def __eq__(self, other):
		try:
			self.assert_equal( other )
			return True
		except AssertionError:
			return False
	
	def _assert_J(self, J):
		self._assert_integer(  dict(J=J)  )
		self._assert_greater(  dict(J=J), 0  )

	def _assert_Q(self, Q):
		self._assert_integer(  dict(Q=Q)  )
		self._assert_greater(  dict(Q=Q), 1  )


	def _assert_array1d(self, d):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert isinstance(value, np.ndarray), s + ' must be a numpy array.'
			assert value.ndim == 1, s + 'must be a one-dimensional array.\nAcutal dimensionality: %d' %value.ndim
			assert value.size > 1, s + 'must have more than one element.'
	
	def _assert_bool(self, d):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert isinstance(value, bool), s + 'must be True or False.'

	def _assert_bounds(self, d, x0, x1, ge=False, le=False):
		for name,value in d.items():
			s  = self._error_prefix(name)
			b0 = (value >= x0) if ge else (value > x0)
			b1 = (value <= x1) if le else (value < x1)
			s0 = 'or equal to ' if ge else ''
			s1 = 'or equal to ' if le else ''
			assert b0 and b1, s + 'must be greater than %s%d and less than %s%d.' %(s0,x0,s1,x1)

	def _assert_callable(self, d, nargs=None):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert callable(value), s + 'must be a callable function.'
			if nargs is not None:
				assert len( inspect.getargspec(value).args )==nargs, s + 'must accept %d input arguments' %nargs

	def _assert_function_result(self, d, inputs=(), shape=None):
		for name,value in d.items():
			s = self._error_prefix(name)
			y = value(*inputs)
			try:
				self._assert_instance( {name:y}, [np.ndarray])
			except AssertionError:
				raise AssertionError(s + 'must return a numpy array.')
			if shape is not None:
				assert(y.shape==shape), s + 'must return a numpy array with shape: %s.\nShape of output from "%s" is: %s' %(str(shape), name, str(y.shape))

	def _assert_greater(self, d, x):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert value > x, s + 'must be greater than %d.' %x

	def _assert_greater_or_equal(self, d, x):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert value >= x, s + 'must be greater than or equal to %d.' %x

	def _assert_greater_than_other(self, d1, d0):
		name0,value0 = list( d0.items() )[0]
		name1,value1 = list( d1.items() )[0]
		s            = self._error_prefix(name1)
		assert value1 > value0, s +  'must be greater than "%s"' %name0


	def _assert_instance(self, d, classes):
		for name,value in d.items():
			s  = self._error_prefix(name)
			if len(classes)==1:
				s2 = 'must be an instance of %s' %(classes[0].__name__)
			else:
				s2 = str(     tuple(  [c.__name__ for c in classes]  )     )
				s2 = 'must be an instance of one of: %s' %s2
			assert isinstance(value, tuple(classes)), s + s2

	def _assert_instance_all(self, d, classes):
		for name,value in d.items():
			s  = self._error_prefix(name)
			if len(classes)==1:
				s2 = 'must contain only instances of %s' %(classes[0].__name__)
			else:
				s2 = str(     tuple(  [c.__name__ for c in classes]  )     )
				s2 = 'must contain only instances from one of: %s' %s2
			for x in value:
				assert isinstance(x, tuple(classes)), s + s2

	def _assert_integer(self, d):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert isinstance(value, int), s + 'must be an integer.'

	def _assert_less(self, d, x):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert value < x, s + 'must be less than %d.' %x
	
	def _assert_list(self, d):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert isinstance(value, (list,tuple)), s + 'must be a list or tuple.'

	def _assert_less_or_equal(self, d, x):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert value <= x, s + 'must be greater than or equal to %d.' %x
	
	def _assert_not_inf_nan(self, d):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert not np.isinf(value), s + 'must not be infinite (np.inf).'
			assert not np.isnan(value), s + 'must not be np.nan.'
		

	def _assert_one_of_two_none(self, d):
		names    = d.keys()
		_none    = [x is None for x in d.values()]
		assert (_none[0] != _none[1]), 'power1d error:  One of the inputs ("%s" and "%s") must have a value and the other must be None.' %tuple(names)
	
	def _assert_same_shape(self, d0, d1, withJ=False):
		name0,model0  = list( d0.items() )[0]
		name1,model1  = list( d1.items() )[0]
		s = self._error_prefix(name0)
		if withJ:
			assert model0.J == model1.J, s + 'must have the same size as "%s"\n--%s: J=%d\n--%s: J=%d' %(name1, name0, model0.J, name1, model1.J)
		assert model0.Q == model1.Q, s + 'must have the same size as "%s"\n--%s: Q=%d\n--%s: Q=%d' %(name1, name0, model0.Q, name1, model1.Q)

	def _assert_same_shape_all(self, d, withJ=False):
		listname,models = list( d.items() )[0]
		models          = list(models)
		s               = self._error_prefix(listname)
		if withJ:
			J           = np.array( [m.J  for m in models] )
			assert np.all(J==J[0]), s + "must contain same-shaped models.\nModels' J values are: %s" %J
		Q           = np.array( [m.Q  for m in models] )
		assert np.all(Q==Q[0]), s + "must contain same-shaped models.\nModels' Q values are: %s" %Q



	def _assert_scalar(self, d):
		for name,value in d.items():
			s = self._error_prefix(name)
			assert np.isscalar(value), s + 'must be a scalar.'
		self._assert_not_inf_nan( d )
	

	def _assert_window(self, d0, d1, x0, x1, asint=True, ge=True, le=True):
		dboth        = d0.copy()
		dboth.update(d1)
		### integer or float:
		if asint:
			self._assert_integer( dboth )
		else:
			self._assert_scalar( dboth )
		self._assert_not_inf_nan( dboth )
		### > or >=
		if ge:
			self._assert_greater_or_equal( dboth, x0 )
		else:
			self._assert_greater( dboth, x0 )
		### < or <=
		if le:
			self._assert_less_or_equal( dboth, x1 )
		else:
			self._assert_less( dboth, x1 )
		### compare values:
		self._assert_greater_than_other(d1, d0)


	def _build(self):
		self.value  = np.zeros(self.Q)   #build a null signal by default
	
	def _error_prefix(self, varname):
		return '\n-----power1d error-----\nClass: "%s"\nInput argument "%s" ' %(self.__class__.__name__, varname)
	
	
	def assert_equal(self, other, tol=1e-6):
		import pytest
		assert self.value  == pytest.approx(other.value,  abs=tol)
	
	def copy(self):
		from copy import deepcopy
		return deepcopy(self)
	
	def dump(self, fpath):
		import pickle
		with open(fpath, 'wb') as f:
			pickle.dump(self, f)
	
	def plot(self, ax=None, q=None, *args, **kwdargs):
		from . _plot import DataPlotter
		plotter = DataPlotter(ax)
		return plotter.plot( q, self.value, *args, **kwdargs )
	
	def toarray(self):
		return self.value.copy()



class _Continuum1D(_ContniuumObject):
	'''
	Abstract class (parent to baselines, signals and noise)
	'''


	def __init__(self, Q=101):
		self._assert_Q(Q)
		self.Q      = Q            #continuum size (number of nodes)
		self.value  = None         #continuum (to be built)
		self._build()              #build the continuum using self.params


	def _init_new(self, other):
		self._check_other(other)
		return DerivedContinuum1D( np.zeros(self.Q) )

	def _check_other(self, other):
		self._assert_instance( dict(other=other), [_Continuum1D])
		assert self.Q==other.Q, 'Continuum lengths must be the same length (base primitive: Q=%d, input primitive: Q=%d).' %(self.Q, other.Q)
	
	def __add__(self, other):
		new       = self._init_new(other) 
		new.value = self.value + other.value
		return new
	def __div__(self, other):
		new       = self._init_new(other) 
		x         = other.value
		x[x==0]   = eps
		new.value = self.value / x
		return new
	def __mul__(self, other):
		new       = self._init_new(other) 
		new.value = self.value * other.value
		return new
	def __pow__(self, other):
		new       = self._init_new(other) 
		new.value = self.value ** other.value
		return new
	def __sub__(self, other):
		new       = self._init_new(other) 
		new.value = self.value - other.value
		return new

	def _build(self):
		self.value  = np.zeros(self.Q)   #build a null signal by default

	def assert_equal(self, other, tol=1e-6):
		import pytest
		assert isinstance(other, _Continuum1D)
		assert self.Q == other.Q
		assert self.value  == pytest.approx(other.value,  abs=tol)
	
	# def dump(self, fpath):
	# 	import pickle
	# 	with open(fpath, 'wb') as f:
	# 		pickle.dump(self, f)

	# def dump(self, fpath):
	# 	from gzip import GzipFile
	# 	import pickle
	# 	with GzipFile(fpath, 'wb') as f:
	# 		pickle.dump(self, f)
	
	
	
	def fliplr(self):
		self.value = self.value[-1::-1]
	def flipud(self, datum=0):
		self.value = -(self.value-datum) + datum


	def plot(self, ax=None, q=None, *args, **kwdargs):
		from . _plot import DataPlotter
		plotter = DataPlotter(ax)
		return plotter.plot( q, self.value, *args, **kwdargs )
		
		
	def save(self, fpath):
		from gzip import GzipFile
		with GzipFile( fpath, 'w') as f:
			np.save(file=f, arr=self.value)

	def set_attr(self, attr, value):
		setattr(self, attr, value)
		self._build()

	def set_attrs(self, **kwargs):
		for attr,value in kwargs.items():
			setattr(self, attr, value)
		self._build()


class DerivedContinuum1D(_Continuum1D):
	'''
	One-dimensional continuum geometry derived from the combination
	of one or more geometric primitives from **power1d.geom**.

	DerivedContinuum1D objects should not be instantiated directly by the user.
	
	Instead they should be instatiated using operators (+ - * / **) to
	combine geometric primitives.
	'''
	def __init__(self, value):
		self._assert_array1d( dict(value=value) )
		super().__init__(value.size)
		self.value = value



class Sample1D(_ContniuumObject):
	'''
	Sample of one or more 1D continua
	'''
	def __init__(self, J=1, Q=101):
		self._assert_J(J)
		self._assert_Q(Q)
		self.J      = J            #sample size
		self.Q      = Q            #continuum size (number of nodes)
		self.value  = None         #continuum (to be built)
		# self._build()              #build the continuum using self.params

	def assert_equal(self, other, tol=1e-6):
		import pytest
		assert isinstance(other, Sample1D)
		assert self.J == other.J
		assert self.Q == other.Q
		assert self.value  == pytest.approx(other.value,  abs=tol)
	
	
	def plot(self, ax=None, q=None, *args, **kwdargs):
		from . _plot import DataPlotter
		plotter = DataPlotter(ax)
		return plotter.plot( q, self.value.T, *args, **kwdargs )

	def set_sample_size(self, J):
		self._assert_J( J )
		self.J      = J

	def set_value(self, y):
		self.value  = y




