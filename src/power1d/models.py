'''
High-level classes for assembling data models and simulating experiments.
'''

# Copyright (C) 2023  Todd Pataky



from copy import deepcopy
import sys,time
from math import ceil
import numpy as np
### import internal functions and classes:
from . _base import _Continuum1D
from . geom import Null
from . noise import _Noise
from . results import SimulationResults




def datasample_from_array( y ):
	'''
	Convenience function for creating a data sample from
	a set of of 1D observations.
	
	The input array must have a shape (J,Q) where:
	
		J = number of observations
		
		Q = number of continuum nodes
	
	WARNING! "datasample_from_array" uses a relatively simple
	SmoothGaussian noise model, and this may NOT embody all
	features of real, experimental noise. In this case more
	complex noise modeling (e.g. Additive, Mixture) may
	be required. Refer to the noise module for more details.
	
	.. plot::
		:include-source:

		import numpy as np
		import matplotlib.pyplot as plt
		import power1d

		y        = power1d.data.weather()['Atlantic']
		model    = power1d.models.datasample_from_array( y )

		plt.close('all')
		fig,axs = plt.subplots(1, 3, figsize=(10,3), tight_layout=True)
		axs[0].plot( y.T )
		np.random.seed(0)
		model.random()
		model.plot( ax=axs[1])
		model.random()
		model.plot( ax=axs[2] )
		labels  = 'Original data', 'DataSample model', 'DataSample model (new noise)'
		[ax.set_title(s) for ax,s in zip(axs,labels)]
		plt.show()
	'''
	from . import geom
	from . noise import from_residuals
	J,Q      = y.shape
	m        = y.mean( axis=0 )
	r        = y - m
	baseline = geom.Continuum1D( m )
	signal   = geom.Null( Q=Q )
	noise    = from_residuals( r )
	model    = DataSample(baseline, signal, noise, J=J)
	return model




class ProgressBar(object):
	'''
	A progress bar for reporting simulation progress to the terminal.
	
	Thank you ChristopheD!!
	http://stackoverflow.com/questions/3160699/python-progress-bar#3160819
	'''
	def __init__(self, width=50, iterations=100):
		sys.stdout.write("[Simulating%s]" % (" " * width))
		sys.stdout.flush()
		sys.stdout.write("\b" * (width+1)) # return to start of line, after '['
		self.update_interval  = float(iterations) / width
		self.i0               = -self.update_interval
	def destroy(self):
		sys.stdout.write("\n\n")
	def update(self, i):
		if (i -self.i0) > self.update_interval:
			sys.stdout.write(".")
			sys.stdout.flush()
			self.i0          += self.update_interval





class DataSample(_Noise):
	'''
	Data Sample model.
	
	Arguments:

	*baseline* ---- a power1d.geom object representing the sample datum
	
	*signal* ---- a power1d.geom object representing the sample signal
	
	*noise* ---- a power1d.noise object representing the sample noise
	
	*J* ---- sample size (int, default 8)
	
	*regressor* ---- an optional regressor that will scale the signal across the *J* observations (length-J numpy array)
	
	Example:
	
	.. plot::
		:include-source:

		import matplotlib.pyplot as plt
		import power1d

		J        = 20
		Q        = 101
		baseline = power1d.geom.Null( Q=Q )
		signal   = power1d.geom.GaussianPulse( Q=Q, q=40, fwhm=15, amp=3.5 )
		noise    = power1d.noise.SmoothGaussian( J=J, Q=Q, mu=0, sigma=1.0, fwhm=20 )
		model    = power1d.models.DataSample(baseline, signal, noise, J=J)
		plt.close('all')
		model.plot(color="g", lw=5)
	'''
	def __init__(self, baseline, signal, noise, J=8, regressor=None):
		### initialize attributes:
		self.J            = None     #: sample size
		self.Q            = None     #: continuum size
		self.baseline     = None     #: baseline model
		self.noise        = None     #: noise model
		self.signal       = None     #: signal model
		# self.value0       = None     #: value in the absence of noise (baseline + signal)
		self.regressor    = None     #: (J,) numpy array representing regressor
		### construct instance:
		self.set_baseline(baseline)
		self.set_signal(signal)
		self.set_noise(noise)
		self.set_sample_size(J)
		self.set_regressor(regressor)
		super().__init__(J, baseline.Q)
		self.random()


	@property
	def hasregressor(self):
		return self.regressor is not None
		
	@property
	def value0(self):  # value in the absence of noise (baseline + signal)
		if self.baseline is None:
			return None
		elif self.signal is None:
			return self.baseline.value
		else:
			return self.baseline.value + self.signal.value
		
	
	def _random(self, subtract_baseline=False):
		self.noise.random()
		if self.hasregressor:
			svalue       = np.array([x * self.signal.value  for x in self.regressor])
		else:
			svalue       = self.signal.value
		bvalue           = 0 if subtract_baseline else self.baseline.value
		self.value       = bvalue + svalue + self.noise.value
		

	def _set_other_value(self):
		self.other.noise.value    = self.noise.value

	
	def copy(self):
		return deepcopy(self)
	
	def get_baseline(self):
		'''
		Return the DataSample's baseline object
		'''
		return self.baseline
	def get_noise(self):
		'''
		Return the DataSample's noise object
		'''
		return self.noise
	def get_signal(self):
		'''
		Return the DataSample's signal object
		'''
		return self.signal


	def link(self, other):
		'''
		Link noise to another DataSample object so that the linked object's
		noise is equivalent to master object's noise.
		'''
		super().link(other)
		self.noise.link(other.noise)


	def plot(self, ax=None, with_noise=True, color='k', lw=3, q=None):
		'''
		Plot an instantaneous representation of the DataSample
		
		Arguments:
		
		*ax* ---- pyplot axes object (default None)
		
		*with_noise* ---- whether or not to plot the noise (bool, default True)
		
		*color* ---- a matplotlib color for all lines (baseline, signal and noise) (default "k" -- black)
		
		*lw* ---- linewidth for the signal (int, default 3)
		
		*q* ---- optional continuum position values (length-Q array of monotonically increasing values) (default None)
		
		
		Example:
		
		.. plot::
			:include-source:

			import matplotlib.pyplot as plt
			import power1d

			J        = 8
			Q        = 101
			baseline = power1d.geom.Null( Q=Q )
			signal   = power1d.geom.GaussianPulse( Q=Q, q=75, fwhm=15, amp=5.0 )
			noise    = power1d.noise.Gaussian( J=J, Q=Q, mu=0, sigma=1.0 )
			model    = power1d.models.DataSample(baseline, signal, noise, J=J)
			plt.close('all')
			model.plot(color="b", lw=5)
		'''
		from . _plot import DataPlotter
		plotter = DataPlotter(ax)
		if self.hasregressor:
			if with_noise:
				from . _plot import _get_colors
				colors = _get_colors(self.J, 'magma')
				for value,c in zip(self.value, colors):
					plotter.plot( q, value, color=c, lw=1 )
		else:
			if with_noise:
				plotter.plot( q, self.value.T, color=color, lw=0.5 )
		plotter.plot( q, self.value0, color=color, lw=lw )


	def random(self, subtract_baseline=False):
		'''
		Generate a random data sample based on the baseline, signal and noise models.
		The value will be stored in the "value" attribute.
		'''
		if self.islinked:
			if self.ismaster:
				self._random(subtract_baseline)
				self._set_other_value()
		else:
			self._random(subtract_baseline)


	def set_baseline(self, baseline):
		'''
		Change the DataSample's baseline object
		
		Arguments:

		*baseline* ---- a power1d.geom object representing the sample datum
		'''
		self._assert_instance( dict(baseline=baseline), [_Continuum1D])
		if self.noise is not None:
			assert baseline.Q == self.noise.Q, 'Baseline and noise must be the same length (baseline: Q=%d, noise: Q=%d).' %(baseline.Q, noise.Q)
		self.baseline = baseline
		self.Q        = baseline.Q
		# self.value0   = baseline.value
		
	def set_noise(self, noise):
		'''
		Change the DataSample's baseline object
		
		Arguments:

		*baseline* ---- a power1d.noise object representing the sample datum
		'''
		assert isinstance(noise, _Noise), 'noise must be an instance of a class from power1d.noise'
		if self.baseline is not None:
			assert noise.Q == self.baseline.Q, 'Baseline and noise must be the same length (baseline: Q=%d, noise: Q=%d).' %(self.baseline.Q, noise.Q)
		self.noise    = noise.copy()
	
	def set_regressor(self, x):
		'''
		Set or change the DataSample's regressor
		
		Arguments:

		*x* ---- a length-J numpy array (float)
		'''
		# if x is None:
		# 	self.hasregressor = False
		# 	self.regressor    = None
		# else:
		# 	self.hasregressor = True
		# 	self.regressor    = x
		self.regressor  = x
			
	
	def set_sample_size(self, J):
		'''
		Change the sample size
		
		Arguments:

		*J* ---- sample size (positive int)
		'''
		super().set_sample_size(J)
        # assert (J >= 3) and (J<=100), 'Sample size (J) must be an integer in the range: (3, 100)'
		assert (J >= 3), 'Sample size (J) must be 3 or greater'
		self.noise.set_sample_size(J)
		


	def set_signal(self, signal):
		'''
		Change the DataSample's signal object
		
		Arguments:

		*signal* ---- a power1d.geom object representing the sample signal
		'''
		if signal is None:
			signal    = Null(self.Q)
		self._assert_instance( dict(signal=signal), [_Continuum1D])
		assert signal.Q == self.baseline.Q, 'Baseline and signal must be the same length (baseline: Q=%d, signal: Q=%d).' %(self.baseline.Q, signal.Q)
		self.signal   = signal
		# self.value0   = self.baseline.value + self.signal.value
		# if self.hasregressor:
		# 	svalue       = [x * self.signal.value  for x in self.regressor]
		# 	self.value0r = np.array( svalue )
		# else:
		# 	svalue = self.signal.value
		





class Experiment(object):
	'''
	Experiment model.
	
	Arguments:

	*data_sample_models* ---- a list of DataSample objects
	
	*fn* ---- a function for computing the test statistic (must accept a sequence of (J,Q)
	arrays and return a length-Q array representing the test statistic continuum). Standard
	test statistic functions are available in **power1d.stats**
	
	Example:
	
	.. plot::
		:include-source:

		import matplotlib.pyplot as plt
		import power1d

		J        = 10
		Q        = 101
		baseline = power1d.geom.Null( Q=Q )
		signal0  = power1d.geom.Null( Q=Q )
		signal1  = power1d.geom.GaussianPulse( Q=Q, q=40, fwhm=15, amp=3.5 )
		noise    = power1d.noise.Gaussian( J=J, Q=Q, mu=0, sigma=1.0 )
		model0   = power1d.models.DataSample(baseline, signal0, noise, J=J)
		model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)
		emodel   = power1d.models.Experiment( [model0, model1], fn=power1d.stats.t_2sample )
		plt.close('all')
		emodel.plot()
	'''
	def __init__(self, data_sample_models, fn):
		### check data_sample_models type
		dmodels  = data_sample_models
		if not isinstance(dmodels, list):
			dmodels = [dmodels]
		for i,m in enumerate(dmodels):
			assert isinstance(m, DataSample), 'all data_sample_models must be instances of power1d.DataSample. data_sample_models[%d] is not.' %i
		### check continuum sizes:
		Q        = dmodels[0].Q
		for i,m in enumerate(dmodels):
			assert (m.Q == Q), 'all data_sample_models must have the same continuum size.  model[0]: Q=%d, model[%d]: Q=%d' %(Q, i, m.Q)
		### check test statistic function:
		assert callable(fn), 'fn must be a callable function'
		### check for errors:
		values  = [m.value for m in dmodels]
		try:
			fn( *values )
		except:
			raise( ValueError('"fn" exited with errors.  It must accept data_model.value as its input argument and return an array with length Q')  )
		### check function output size:
		y   = fn( *values )
		assert isinstance(y, np.ndarray), '"fn" must return a numpy array.'
		assert y.ndim==1, '"fn" must return a one-dimensional numpy array.'
		assert y.size==Q, '"fn" must return a numpy array of length Q.'
		### set attributes:
		self.Q           = Q              #: continuum length
		self.data_models = dmodels        #: data models
		self.nmodels     = len(dmodels)   #: number of data models
		self.fn          = fn             #: test statistic function
		self.Z           = None           #: output test statistic continua
		### copy models (to ensure that each data model has different noise):
		self.data_models = [m.copy() for m in self.data_models]


	def __eq__(self, other):
		try:
			self.assert_equal( other )
			return True
		except AssertionError:
			return False
	
	
	def assert_equal(self, other, tol=1e-6):
		import pytest
		assert isinstance(other, Experiment)
		assert self.Q == other.Q
		for dmodel,dmodel0 in zip(self.data_models, other.data_models):
			assert dmodel == dmodel0
		# assert self.Z0  == pytest.approx(other.value,  abs=tol)
	
	
	def plot(self, ax=None, with_noise=True, colors=None, q=None):
		'''
		Plot an instantaneous representation of the Experiment model.
		
		Arguments:
		
		*ax* ---- pyplot axes object (default None)
		
		*with_noise* ---- whether or not to plot the noise (bool, default True)
		
		*colors* ---- an optional sequence of matplotlib colors, one for each DataSample (default None)
		
		*q* ---- optional continuum position values (length-Q array of monotonically increasing values) (default None)
		
		
		Example:
		
		.. plot::
			:include-source:

			import numpy as np
			import matplotlib.pyplot as plt
			import power1d

			J        = 8
			Q        = 101
			baseline = power1d.geom.Null( Q=Q )
			signal0  = power1d.geom.Null( Q=Q )
			signal1  = power1d.geom.GaussianPulse( Q=Q, q=40, fwhm=15, amp=3.5 )
			noise    = power1d.noise.Gaussian( J=J, Q=Q, mu=0, sigma=1.0 )
			model0   = power1d.models.DataSample(baseline, signal0, noise, J=J)
			model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)
			emodel   = power1d.models.Experiment( [model0, model1], fn=power1d.stats.t_2sample )
			plt.close('all')
			emodel.plot( colors=("k", "r"), q=np.linspace(0, 1, Q) )
		'''
		if colors is None:
			from . _plot import _get_colors
			colors       = _get_colors( self.nmodels )
		for m,c in zip(self.data_models, colors):
			m.plot(ax, with_noise, c, q=q)


	def simulate_single_iteration(self):
		'''
		Simulate a single experiment.
		
		Calling this method will generate new data samples, one for each DataSample model,
		and will then compute and return the test statistic continuum associated with
		the new data samples.
		'''
		for m in self.data_models:
			m.random(subtract_baseline=True)
		values       = [m.value for m in self.data_models]
		return self.fn( *values )


	def simulate(self, iterations=1000, progress_bar=True):
		'''
		Iteratively simulate a number of experiments.
		
		Calling this method will repeatedly call **simulate_single_iteration** and
		will store the resulting continua in the **Z** attribute;  if 1000 iterations
		are executed then **Z** will be a (1000,Q) array when simulation is complete.
		
		Arguments:
		
		*iterations* ---- number of iterations to perform (positive int, default 1000)
		
		*progress_bar* ---- whether or not to display a progress bar in the terminal (bool, default True)
		
		Example:
		
		.. plot::
			:include-source:

			import numpy as np
			import matplotlib.pyplot as plt
			import power1d

			J        = 8
			Q        = 101
			baseline = power1d.geom.Null( Q=Q )
			signal0  = power1d.geom.Null( Q=Q )
			signal1  = power1d.geom.GaussianPulse( Q=Q, q=40, fwhm=15, amp=3.5 )
			noise    = power1d.noise.Gaussian( J=J, Q=Q, mu=0, sigma=1.0 )
			model0   = power1d.models.DataSample(baseline, signal0, noise, J=J)
			model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)
			emodel   = power1d.models.Experiment( [model0, model1], fn=power1d.stats.t_2sample )
			
			emodel.simulate(iterations=50, progress_bar=True)
			plt.close('all')
			plt.plot( emodel.Z.T, color='k', lw=0.5 )
			plt.title('Test statistic continua')
		'''
		_msg             = 'iterations must be an integer between 10 and 100,000'
		assert isinstance(iterations, int), _msg
		assert (iterations>=10) and (iterations<=1e5), _msg
		### initialize progress bar
		# print 'Simulating...'
		if progress_bar:
			pbar         = ProgressBar(width=40, iterations=iterations)
		Z                = []
		for i in range(iterations):
			if progress_bar:
				pbar.update(i)
			z            = self.simulate_single_iteration()
			Z.append(z)
		### clean up:
		if progress_bar:
			pbar.destroy()
		for m in self.data_models:
			m.random()   #set new random values (no baseline subtracting)
		self.Z           = np.array(Z)
		
	def set_sample_size(self, JJ):
		'''
		Change the sample sizes of the models
		
		Arguments:

		*J* ---- sample size (positive int or list of positive int)
		'''
		JJ = [JJ] if isinstance(JJ, int) else list(JJ)
		for m,J in zip(self.data_models, JJ):
			m.set_sample_size(J)





class ExperimentSimulator(object):
	'''
	Two-experiment simulator.
	
	This is a convenience class for simulating two experiments:
	
	- "null" experiment:  represents the null hypothesis
	- "effect" experiment:  represents the alternative hypothesis
	
	Simulation results will all share the following characteristics:
	
	- Alpha-based critical thresholds are be computed for the "null" experiment
	- Power calculations use those thresholds to conducted power analysis on the "effect" experiment.
	
	Arguments:
	
	*model0* ---- the "null" experiment model (power1d.models.Experiment)
	*model1* ---- the "effect" experiment model (power1d.models.Experiment)
	
	.. note:: If the "null" and "effect" models are identical then power will be alpha (or numerically close to it) by definition. 
	'''
	def __init__(self, model0, model1):
		for m in [model0, model1]:
			assert isinstance(m, Experiment), 'Both inputs to ExperimentSimulator must be instances of power1d.Experiment'
		self.Q        = model0.Q   #: continuum size
		self.model0   = model0     #: "null" experiment model
		self.model1   = model1     #: "effect" experiment model


	def load_simulation_results(self, filename):
		'''
		Load saved simulation results.
		
		Arguments:
		
		*filename* ---- full path to saved results file
		
		Outputs:
		
		*results* ---- a results object (power1d.results.SimulationResults)
		
		
		.. note:: Simulating experiments with large sample sizes, complex\
		DataSample models and/or a large number of simulation iterations can\
		be computationally demanding. The **load_simulation_results** method\
		allows you to probe simulation results efficiently, by saving the\
		results to disk following simulation, then subsequently skipping the\
		"simulate" command. In the example below the commands between the\
		"###" flags should be commented after executing "simulate" once.
		
		Example:
		
		.. plot::
			:include-source:

			import numpy as np
			import matplotlib.pyplot as plt
			import power1d

			J        = 8
			Q        = 101
			baseline = power1d.geom.Null( Q=Q )
			signal0  = power1d.geom.Null( Q=Q )
			signal1  = power1d.geom.GaussianPulse( Q=Q, q=40, fwhm=15, amp=1.2 )
			noise    = power1d.noise.Gaussian( J=J, Q=Q, mu=0, sigma=1.0 )
			model0   = power1d.models.DataSample(baseline, signal0, noise, J=J)
			model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)
			
			emodel0  = power1d.models.Experiment( [model0, model0], fn=power1d.stats.t_2sample )
			emodel1  = power1d.models.Experiment( [model0, model1], fn=power1d.stats.t_2sample )
		
			
			###  COMMANDS BELOW SHOULD BE COMMENTED AFTER EXECUTING ONCE
			sim      = power1d.models.ExperimentSimulator(emodel0, emodel1)
			results  = sim.simulate(iterations=200, progress_bar=True)
			fname    = "/tmp/results.npz"
			results.save( fname )
			###  COMMENTS ABOVE SHOULD BE COMMENTED AFTER EXECUTING ONCE
			
			#Then the results can be re-loaded:
			fname         = "/tmp/results.npz"
			saved_results = sim.load_simulation_results( fname )
			plt.close('all')
			saved_results.plot()
		'''
		with np.load(filename) as D:
			Z0,Z1,dt  = D['Z0'], D['Z1'], D['dt']
		return SimulationResults( self.model0, self.model1, Z0, Z1, dt )


	def sample_size(self, power=0.8, alpha=0.05, niter0=200, niter=1000, coi=None):
		'''
		Estimate sample size required to achieve a target power.
		
		When *coi* is None the omnibus power is used.
		When *coi* is a dictionary then the COI power is used, e.g. :code:`coi=dict(q=65, r=3)`

		Arguments:
		
		*power* ---- target power (default: 0.8, range: [0,1])
		
		*alpha* ---- Type I error rate (default: 0.05, range: [0,1])
		
		*niter0* ---- Number of iterations for initial, approximate solution (default: 200, range: [200,])
		
		*niter* ---- Number of iterations for final solution (default: 1000, range: [1000,])
		
		*coi* ---- Center-of-interest (default: None;  either None or dict(q=q, r=r)  where q is the COI and r is its radius; e.g. coi=dict(q=65, r=3))
		
		Outputs:
		
		Dictionary containing:
		
		*nstar* ---- estimated sample size required to achieve the target power
		
		*n* --- array of sample sizes used for the final power calculation
		
		*p* --- array of final power values
		'''
		def sim_single(n, niter=200):
			self.model0.set_sample_size( n )
			self.model1.set_sample_size( n )
			results = self.simulate( niter, progress_bar=False )
			results.set_alpha( alpha )
			if coi is None:
				p   = results.p_reject1
			else:
				results.set_coi( (coi['q'], coi['r']) )
				p   = results.p_coi1[0]
			return p
		# approximate solution:
		n,p         = 3, 0
		data_approx = []
		while True:
			p       = sim_single(n, niter=niter0)
			data_approx.append( (n,  p) )
			if p > power:
				break
			else:
				n  += 1
		# finer detail in area of approximate solution:
		ns    = list( range( n-3, n+3 ) )
		ps    = np.array(  [sim_single( nn, niter=niter )  for nn in ns]  )
		ind   = np.argwhere( ps > power ).ravel()[0]
		nstar = ns[ind]
		return dict(nstar=nstar, n=np.array(ns), p=ps, target_power=power, alpha=alpha, coi=coi)
	
	
	def simulate(self, iterations=50, progress_bar=True, two_tailed=False, _qprogressbar=None):
		'''
		Iteratively simulate a number of experiments.
		
		Calling this method will repeatedly call **simulate_single_iteration** for
		both the "null" model and the "effect" model.
		
		Arguments:
		
		*iterations* ---- number of iterations to perform (positive int, default 1000)
		
		*progress_bar* ---- whether or not to display a progress bar in the terminal (bool, default True)

		*two_tailed* ---- whether or not to use two-tailed inference (bool, default False)
		
		Outputs:
		
		*results* ---- a results object (power1d.results.SimulationResults)
		
		
		Example:
		
		.. plot::
			:include-source:

			import numpy as np
			import matplotlib.pyplot as plt
			import power1d

			J        = 8
			Q        = 101
			baseline = power1d.geom.Null( Q=Q )
			signal0  = power1d.geom.Null( Q=Q )
			signal1  = power1d.geom.GaussianPulse( Q=Q, q=40, fwhm=15, amp=1.0 )
			noise    = power1d.noise.Gaussian( J=J, Q=Q, mu=0, sigma=0.3 )
			model0   = power1d.models.DataSample(baseline, signal0, noise, J=J)
			model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)
			
			emodel0  = power1d.models.Experiment( [model0, model0], fn=power1d.stats.t_2sample )
			emodel1  = power1d.models.Experiment( [model0, model1], fn=power1d.stats.t_2sample )
		
			sim      = power1d.models.ExperimentSimulator(emodel0, emodel1)
			results  = sim.simulate(iterations=200, progress_bar=True)
			
			plt.close('all')
			results.plot()
		'''
		_msg          = 'iterations must be an integer greater than or equal to 50'
		assert isinstance(iterations, int), _msg
		assert iterations >= 50, _msg
		if progress_bar:
			pbar      = ProgressBar(width=40, iterations=iterations)
		Z0,Z1         = [],[]
		t0            = time.time()
		for i in range(iterations):
			if progress_bar:
				pbar.update(i)
				if _qprogressbar is not None:
					_qprogressbar.update(i)
			Z0.append(  self.model0.simulate_single_iteration()  )
			Z1.append(  self.model1.simulate_single_iteration()  )
		### clean up:
		if progress_bar:
			pbar.destroy()
			if _qprogressbar is not None:
				_qprogressbar.reset()
		for emodel in [self.model0, self.model1]:
			for m in emodel.data_models:
				m.random()   #set new random values (no baseline subtracting)
		### assemble results:
		dt            = time.time() - t0
		Z0,Z1         = np.asarray(Z0), np.asarray(Z1)
		if two_tailed:
			Z0,Z1     = np.abs(Z0), np.abs(Z1)
		return SimulationResults( self.model0, self.model1, Z0, Z1, dt, two_tailed=two_tailed )


