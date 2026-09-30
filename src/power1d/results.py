'''
A module for probing, displaying and plotting **power1d**
simulation results.

This module contains the **SimulationResults** class which
is meant to be instantiated only by **power1d** and not by
the user. However, once instantiated the user can run
a variety of analyses using the methods described below.

Importantly, since simualations can take a long time to run
users are encouraged to save **SimulationResults** objects
using the "save" method and re-loaded using the
**load_simulation_results** method of an
**ExperimentSimulator** object.

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

# Copyright (C) 2023  Todd Pataky



import numpy as np
from scipy import ndimage
from . _base import _Power1DObject
from . roi import RegionOfInterest


def _pkmax(Z, u, k):
	'''
	Compute the probability of observing a cluster with extent *k*.
	
	Z :  (nIterations x nNodes) array of test statistic continua
	u :  threshold (float)
	k :  cluster extent (int)
	'''
	struct = np.array([ [True]*k ])
	H      = ndimage.binary_hit_or_miss( Z > u , structure1=struct)
	p      = np.any(H, axis=1).mean()
	return p


class SimulationResultsSummary(object):
	'''
	A class containing a reduced set of simulation results.
	Designed primarily for unit testing.
	
	Create using SimulationResults.as_summary()
	'''
	def __init__(self, results):
		self.alpha         = results.alpha
		self.dt            = results.dt
		### primary and secondary distributions:
		self.z0            = results.z0
		self.z1            = results.z1
		### user-selected inference parameters:
		self.coi           = results.coi
		self.coir          = results.coir
		self.poi           = results.poi
		self.k             = results.k
		### probability results:
		self.zstar         = results.zstar
		self.p_reject0     = results.p_reject0
		self.p_reject1     = results.p_reject1
		self.p1d_poi0      = results.p1d_poi0
		self.p1d_poi1      = results.p1d_poi1
		self.p1d_coi0      = results.p1d_coi0
		self.p1d_coi1      = results.p1d_coi1
		### probability results (at user-specified POIs and COIs)
		self.p_poi0        = results.p_poi0
		self.p_poi1        = results.p_poi1
		self.p_coi0        = results.p_coi0
		self.p_coi1        = results.p_coi1
	
	def __eq__(self, other):
		try:
			self.assert_equal( other )
			return True
		except AssertionError:
			return False
	
	def __repr__(self):
		s  = ''
		s += '------------------------\n'
		s += 'power1d simulation results SUMMARY\n'
		s += '------------------------\n'
		s += 'Simulation overview\n'
		s += '   number of iterations = %d\n'  %self.niters
		s += '   duration             = %.2f s\n'  %self.dt
		s += '------------------------\n'
		s += 'Main parameters\n'
		s += '   alpha              = %0.5f\n' %self.alpha
		s += '   critical test stat = %.5f\n'  %self.zstar
		s += '------------------------\n'
		s += 'H0 rejection probabilities (omnibus)\n'
		s += '   Null model:    p = %.5f\n' %self.p_reject0
		s += '   Effect model:  p = %.5f\n' %self.p_reject1
		s += '------------------------\n'
		if self.coi is not None:
			s += 'H0 rejection probability (COI)\n'
			s += '   Position   Radius   Null model   Effect model\n'
			for (coi,r),p0,p1 in zip(self.coi, self.p_coi0, self.p_coi1):
				s += '   %s    %s      %0.5f       %0.5f\n' %(str(coi).rjust(5, ' '), str(r).rjust(5, ' '), p0, p1)
			s += '------------------------\n'
		if self.poi is not None:
			s += 'H0 rejection probability (POI)\n'
			s += '   Position     Null model    Effect model\n'
			for poi,p0,p1 in zip(self.poi, self.p_poi0, self.p_poi1):
				s += '   %s        %0.5f       %0.5f\n' %(str(poi).rjust(5, ' '), p0, p1)
			s += '------------------------\n'
		return s
	
	@property
	def niters(self):
		return self.z0.size



	def assert_equal(self, other, tol=1e-6):
		import pytest
		assert isinstance(other, SimulationResultsSummary)
		assert self.z0  == pytest.approx(other.z0,  abs=tol)
		assert self.z1  == pytest.approx(other.z1,  abs=tol)
		assert self.p1d_poi0 == pytest.approx(other.p1d_poi0,  abs=tol)
		assert self.p1d_poi1 == pytest.approx(other.p1d_poi1,  abs=tol)
		assert self.p1d_coi0 == pytest.approx(other.p1d_coi0,  abs=tol)
		assert self.p1d_coi1 == pytest.approx(other.p1d_coi1,  abs=tol)
	

	def dump(self, fpath):
		import pickle
		with open(fpath, 'wb') as f:
			pickle.dump(self, f)



class SimulationResults(_Power1DObject):
	'''
	A class containing **power1d** simulation results
	including distributions and derived probabilities.
	'''
	def __init__(self, model0, model1, Z0, Z1, dt, two_tailed=False, alpha=0.05):
		### experimental models (for plotting only)
		self.model0        = model0  #: the "null" experiment model  (an instance of **power1d.models.Experiment**)
		self.model1        = model1  #: the "alternative" experiment model   (an instance of **power1d.models.Experiment**)
		self.dt            = dt      #: total simulation duration (s)
		### simulated experimental continua (constants):
		self.niters        = Z0.shape[0] #: number of simulation iterations
		self.Q             = Z0.shape[1] #: continuum size
		self.Z0            = Z0      #: test statistic continua ("null" experiment)
		self.Z1            = Z1      #: test statistic continua ("alternative" experiment)
		self.two_tailed    = bool( two_tailed )
		#: inference parameters:
		self.alpha         = alpha   #: Type I error rate (default 0.05)
		self.roi           = None    #: region(s) of interest (default: whole continuum)
		### power-relevant distribution summarizers:
		self.zstar         = None    #: critical threshold for omnibus null (based on Z0max)
		self.z0            = None    #: distribution upon which omnibus H0 rejection decision is based ("null" model)
		self.z1            = None    #: distribution upon which omnibus H0 rejection decision is based ("alternative" model)
		### user-selected inference parameters:
		self.coi           = None    #: continuum centers-of-interest (if any) for which summary powers will be displayed when printing results
		self.coir          = None    #: center-of-interest radius (for coi continuum, default 3)
		self.poi           = None    #: continuum points-of-interest (if any) for which summary powers will be displayed when printing results
		self.k             = None    #: cluster extent (default 1);  clusters in the excursion set with smaller extents will be ignored when computing probabilities
		### probability results:
		self.p_reject0     = None    #: omnibus null rejection probability for the "null" experiment (alpha by defintion)
		self.p_reject1     = None    #: omnibus null rejection probability for the "alternative" experiment (power by defintion)
		self.p1d_poi0      = None    #: POI probability continuum for the "null" experiment
		self.p1d_poi1      = None    #: POI probability continuum for the "alternative" experiment
		self.p1d_coi0      = None    #: COI probability continuum for the "null" experiment
		self.p1d_coi1      = None    #: COI probability continuum for the "alternative" experiment
		### probability results (at user-specified POIs and COIs)
		self.p_poi0        = None
		self.p_poi1        = None
		self.p_coi0        = None
		self.p_coi1        = None
		### set default parameters and calculate probabilities:
		self._init()


	def __eq__(self, other):
		try:
			self.assert_equal( other )
			return True
		except AssertionError:
			return False


	def _to_dict(self):
		'''
		Serialisable form: the simulated distributions and the inference settings.
		The experiment models are included when they can be serialised (None otherwise).
		'''
		def model(m):
			try:
				return None if m is None else m._to_dict()
			except ValueError:
				return None
		full_roi = bool( np.all(self.roi.value) )
		return dict(
			model0     = model(self.model0),
			model1     = model(self.model1),
			Z0         = self.Z0.copy(),
			Z1         = self.Z1.copy(),
			dt         = float(self.dt),
			two_tailed = self.two_tailed,
			alpha      = float(self.alpha),
			roi        = None if full_roi else np.asarray(self.roi.value, dtype=bool).copy(),
			coi        = None if self.coi is None else [[int(q), int(r)] for q, r in self.coi],
			coir       = int(self.coir),
			poi        = None if self.poi is None else [int(x) for x in self.poi],
			k          = int(self.k),
			**{'class': 'SimulationResults'} )


	@classmethod
	def _from_dict_custom(cls, d):
		from . _serial import from_dict
		model0  = None if d.get('model0') is None else from_dict(d['model0'])
		model1  = None if d.get('model1') is None else from_dict(d['model1'])
		r       = cls(model0, model1, np.asarray(d['Z0']), np.asarray(d['Z1']), d['dt'],
		              two_tailed=d['two_tailed'], alpha=d['alpha'])
		r.coir  = d['coir']
		if d['roi'] is not None:
			r.set_roi( np.asarray(d['roi'], dtype=bool) )
		if d['k'] != 1:
			r.set_cluster_size_threshold( d['k'] )
		if d['coi'] is not None:
			r.set_coi( *[tuple(c) for c in d['coi']] )
		if d['poi'] is not None:
			r.set_poi( *d['poi'] )
		r._calculate_probabilities()
		return r


	def __repr__(self):
		s  = ''
		s += '------------------------\n'
		s += 'power1d simulation results\n'
		s += '------------------------\n'
		s += 'Simulation overview\n'
		s += '   number of iterations = %d\n'  %self.niters
		s += '   duration             = %.2f s\n'  %self.dt
		s += '------------------------\n'
		s += 'Main parameters\n'
		s += '   alpha              = %0.5f\n' %self.alpha
		s += '   critical test stat = %.5f\n'  %self.zstar
		s += '------------------------\n'
		s += 'H0 rejection probabilities (omnibus)\n'
		s += '   Null model:    p = %.5f\n' %self.p_reject0
		s += '   Effect model:  p = %.5f\n' %self.p_reject1
		s += '------------------------\n'
		if self.coi is not None:
			s += 'H0 rejection probability (COI)\n'
			s += '   Position   Radius   Null model   Effect model\n'
			for (coi,r),p0,p1 in zip(self.coi, self.p_coi0, self.p_coi1):
				s += '   %s    %s      %0.5f       %0.5f\n' %(str(coi).rjust(5, ' '), str(r).rjust(5, ' '), p0, p1)
			s += '------------------------\n'
		if self.poi is not None:
			s += 'H0 rejection probability (POI)\n'
			s += '   Position     Null model    Effect model\n'
			for poi,p0,p1 in zip(self.poi, self.p_poi0, self.p_poi1):
				s += '   %s        %0.5f       %0.5f\n' %(str(poi).rjust(5, ' '), p0, p1)
			s += '------------------------\n'
		return s



	
	def _init(self):
		self.coir       = 3
		self.roi        = RegionOfInterest(   np.array( [True] * self.Q )   )
		self.k          = 1
		self._calculate_probabilities()




	#----------------------------------------
	# COI-level calculations
	#----------------------------------------

	def _calculate_prob_coi(self):
		### P(H0 reject) at centers of interest:
		if self.coi is not None:
			p            = [self._calculate_prob_coi_single_node(x, r)  for x,r in self.coi]
			p0,p1        = np.array(p).T
			self.p_coi0  = p0
			self.p_coi1  = p1

	def _calculate_prob_coi_single_node(self, x, r):
		x0,x1    = x-r, x+r+1
		x0,xa    = max(x0, 0), min(x1, self.Q)
		if self.k > 1:
			p0       = _pkmax(self.Z0[:,x0:x1], self.zstar, self.k)
			p1       = _pkmax(self.Z1[:,x0:x1], self.zstar, self.k)
		else:
			z0max    = self.Z0[:,x0:x1].max(axis=1)
			z1max    = self.Z1[:,x0:x1].max(axis=1)
			p0       = (z0max>self.zstar).mean()
			p1       = (z1max>self.zstar).mean()
		return p0,p1

	def _calculate_prob_continua_coi(self):
		p0,p1            = [],[]
		for x in range(self.Q):
			pp0,pp1      = self._calculate_prob_coi_single_node(x, self.coir)
			p0.append(  pp0  )
			p1.append(  pp1  )
		self.p1d_coi0    = np.array(p0)
		self.p1d_coi1    = np.array(p1)

	#----------------------------------------
	# POI-level calculations
	#----------------------------------------
	def _calculate_prob_continua_poi(self):
		# P(H0 reject) at individual nodes:
		self.p1d_poi0    = (self.Z0 > self.zstar).mean(axis=0)
		self.p1d_poi1    = (self.Z1 > self.zstar).mean(axis=0)

	def _calculate_prob_poi(self):
		### P(H0 reject) at points of interest:
		if self.poi is not None:
			self.p_poi0  = self.p1d_poi0[self.poi]
			self.p_poi1  = self.p1d_poi1[self.poi]

	#----------------------------------------
	# Omnibus-level calculations
	#----------------------------------------
	def _calculate_critical_threshold(self):
		i                = self.roi.value
		perc             = 100 * (1 - self.alpha)
		### compute distributions of the maximum test statistic:
		self.z0          = self.Z0[:,i].max(axis=1)
		self.z1          = self.Z1[:,i].max(axis=1)
		### compute maximum cluster extent distribution:
		self.zstar       = np.percentile(self.z0, perc)
	
	def _calculate_prob_omnibus(self):
		if self.k > 1:
			self.p_reject0   = _pkmax(self.Z0, self.zstar, self.k)
			self.p_reject1   = _pkmax(self.Z1, self.zstar, self.k)
		else:
			self.p_reject0   = (self.z0 > self.zstar).mean() #should be alpha by definition, this is just a numerical verification
			self.p_reject1   = (self.z1 > self.zstar).mean()

	#----------------------------------------
	# Calculate everything
	#----------------------------------------
	def _calculate_probabilities(self):
		self._calculate_critical_threshold()
		self._calculate_prob_omnibus()
		self._calculate_prob_continua_poi()
		self._calculate_prob_continua_coi()
		self._calculate_prob_poi()
		self._calculate_prob_coi()


	def as_summary(self):
		return SimulationResultsSummary( self )
	
	
	def assert_equal(self, other, tol=1e-6):
		import pytest
		assert isinstance(other, SimulationResults)
		assert self.model0 == other.model0
		assert self.Z0  == pytest.approx(other.Z0,  abs=tol)
		assert self.Z1  == pytest.approx(other.Z1,  abs=tol)
		assert self.z0  == pytest.approx(other.z0,  abs=tol)
		assert self.z1  == pytest.approx(other.z1,  abs=tol)
		assert self.p1d_poi0 == pytest.approx(other.p1d_poi0,  abs=tol)
		assert self.p1d_poi1 == pytest.approx(other.p1d_poi1,  abs=tol)
		assert self.p1d_coi0 == pytest.approx(other.p1d_coi0,  abs=tol)
		assert self.p1d_coi1 == pytest.approx(other.p1d_coi1,  abs=tol)
		
	
	
	def dump(self, fpath):
		import pickle
		with open(fpath, 'wb') as f:
			pickle.dump(self, f)

	# def dump(self, fpath, precision=3):
	# 	from copy import deepcopy
	# 	import pickle
	# 	results = deepcopy( self )
	# 	print( 'with around')
	# 	results.Z0 = np.around(results.Z0, precision)
	# 	results.Z1 = np.around(results.Z1, precision)
	# 	results.z0 = np.around(results.z0, precision)
	# 	results.z1 = np.around(results.z1, precision)
	# 	with open(fpath, 'wb') as f:
	# 		pickle.dump(results, f)
	
	
	def plot(self, q=None):
		'''
		Plot simulation results.
		
		This will plot the "null" and "effect" experiment models along
		with their omnibus powers and their power continua.
		
		By defintion the "null" experiment model will have an omnibus power
		of *alpha* and its power continua will be considerably smaller than
		*alpha*, with power decreasing as a function of continuum size.
		
		The "effect" experiment model will have an omnibus power that
		depends on the signal and noise models contained in its DataSample
		models.
		
		Keyword arguments:
		
		q ------ an optional array specifying continuum points (default None)
		
		
		Example:
		
		>>>  Q = results.Q  #continuum size
		>>>  results.plot( q=np.linspace(0, 1, Q) )
		'''
		from . _plot import _plot_results_one_datamodel, _plot_results_multiple_datamodels
		ndatamodels  = self.model0.nmodels
		hasregressor = self.model1.data_models[0].hasregressor
		if ndatamodels==1 and not hasregressor:
			_plot_results_one_datamodel(self, q)
		else:
			_plot_results_multiple_datamodels(self, q)


	def save(self, filename):
		'''
		Save simulation results.
		
		Arguments:
		
		filename ------ full path to a file. Should have a ".npz" extension (numpy compressed format).  It must follow the rules of **np.savez**.
		
		Example:
		
		>>>  results.save( "/Users/username/Desktop/my_results.npz" )
		'''
		np.savez_compressed(filename, Z0=self.Z0, Z1=self.Z1, dt=self.dt)
	
	
	def set_alpha(self, alpha):
		'''
		Set the Type I error rate.
		
		After calling this method all probabilities will be re-computed automatically.
		
		Arguments:
		
		alpha ------ Type I error rate (float between 0 and 1)
		
		Example:
		
		>>>  results.set_alpha( 0.01 )
		'''
		_msg = 'alpha must be a scalar greater than zero and less than one.'
		assert isinstance(alpha, float), msg
		assert (alpha>0) and (alpha<1), msg
		self.alpha  = alpha
		self._calculate_probabilities()

	def set_cluster_size_threshold(self, k):
		'''
		Set a cluster size threshold (*k*) for null hypothesis (H0) rejection.
		
		By default *k*=1 is used, in which case only the distribution of the
		continuum maximum is used as the H0 rejection criterion. For *k*
		larger than one, the H0 rejection criterion becomes defined by the
		distribution of excursion set (supra-threshold cluster) geometry.
		In particular, all excursion set clusters with extents less than *k*
		are ignored both in critical threshold computations and in power
		computations.
		
		After calling this method all probabilities will be re-computed automatically.
		
		Arguments:
		
		k ------ Type I error rate (integer greater than 0)
		
		Example:
		
		>>>  results.set_cluster_size_threshold( w )
		'''
		assert isinstance(k, int), 'cluster size threshold must be an integer.'
		assert k >=0, 'cluster size threshold must be greater than or equal to zero.'
		assert k < self.Q, 'cluster size threshold radius must be less than the continuum length.'
		self.k      = k
		self._calculate_probabilities()
	
	def set_coi(self, *coi):
		'''
		Set centers-of-interest (COIs) to be displayed when printing.
		
		An arbibtrary number of (location, radius) pairs specifying
		locations of empirical interest. These COI results will be
		displayed when using the "print" command as indicated in the
		example below.
		
		After calling this method all relevant probabilities will be re-computed automatically.
		
		Arguments:
		
		coi ------ A sequence of integer pairs (location, radius)
		
		Example:
		
		>>>  results.set_coi( (10,2), (50,8), (85,4) )
		>>>  print(results)
		'''
		_msg = 'input to set_coi must be a sequence of two-tuples.'
		try:
			coi     = np.array(coi)
		except:
			raise(ValueError(_msg))
		assert coi.ndim==2, _msg
		assert coi.shape[1]==2, _msg
		assert coi.dtype==int, 'input to set_coi must contain only integers'
		### ensure that all coi centers lie in the ROI:
		x,r         = coi.T
		ind         = np.argwhere(self.roi.value).flatten()
		for i,(xx,rr) in enumerate( zip(x,r) ):
			assert xx>=0, 'all centers of interest must be greater than or equal to zero'
			assert xx<=self.Q, 'all centers of interest must be less than or equal to the continuum size'
			assert xx in ind, 'all centers of interest must lie inside the specified region-of-interest.  Point %d is outside the ROI.' %xx
			assert xx-rr in ind, 'no center of interest can extend beyond the specified region-of-interest.  Point %d minus radius %d extends outside the ROI.' %(xx,rr)
			assert xx+rr in ind, 'no center of interest can extend beyond the specified region-of-interest.  Point %d plus radius %d extends outside the ROI.' %(xx,rr)
		self.coi    = coi
		self._calculate_prob_coi()

	def set_coi_radius(self, r):
		'''
		Set centers-of-interest (COI) radius for the COI continuum results.
		
		When using the "plot" method COI continuum results will be displayed
		by default. These indicate power associated with small regions
		surrounding the particular continuum point.
		
		If the continuum radius is small the COI power will generally be
		smaller than the omnibus power. As the COI radius increases the COI
		power will plateau at the omnibus power.
		
		If the continuum radius is one then the COI power continuum is
		equivalent to the point-of-interest (POI) power continuum.
		
		After calling this method all relevant probabilities will be re-computed automatically.
		
		Arguments:
		
		r ------ COI radius (int)
		
		Example:
		
		>>>  results.set_coi_radius( 5 )
		>>>  results.plot()
		'''
		assert isinstance(r, int), 'coi radius must be an integer.'
		assert r >=0, 'coi radius must be greater than or equal to zero.'
		assert r < self.Q, 'coi radius must be less than the continuum length.'
		self.coir    = r
		self._calculate_prob_continua_coi()

	def set_poi(self, *poi):
		'''
		Set points-of-interest (POIs) to be displayed when printing.
		
		The power associated with an arbibtrary number of continuum POIs 
		will be displayed when using the "print" command as indicated
		in the example below.
		
		After calling this method all relevant probabilities will be re-computed automatically.
		
		Arguments:
		
		poi ------ A sequence of integers
		
		Example:
		
		>>>  results.set_coi( (10,2), (50,8), (85,4) )
		>>>  print(results)
		'''
		try:
			self.poi    = list(poi)
		except:
			raise(ValueError('input to set_poi must be a sequence of integers'))
		ind         = np.argwhere(self.roi.value).flatten()
		for i,x in enumerate(self.poi):
			assert isinstance(x, int), 'all points of interest must be integers.  Input #%d is not an integer.' %i
			assert x>=0, 'all points of interest must be greater than or equal to zero'
			assert x<=self.Q, 'all points of interest must be less than or equal to the continuum size'
			assert x in ind, 'all points of interest must lie inside the specified region-of-interest.  Point %d is outside the ROI.' %x
		self._calculate_prob_poi()

	def set_roi(self, roi):
		'''
		Set region of interest (ROI).
		
		An ROI constrains the continuum extent of the null hypothesis (H0).
		By default the entire continuum is considered to be the ROI of
		interest. Single or multiple ROIs can be specified as indicated
		below, and this will cause all data outside the ROIs to be
		ignored.
		
		Setting the ROI to a single point will yield results associated
		with typical power calculations. That is, a single continuum point
		behaves the same as a single scalar dependent variable.
		
		After calling this method all probabilities will be re-computed automatically.
		
		Arguments:
		
		roi ------ a boolean array or a **power1d.continua.RegionOfInterest** object
		
		Example:
		
		>>>  Q = results.Q  #continuum size
		>>>  roi = np.array( [False]*Q )  #boolean array
		>>>  roi[50:80] = True
		>>>  results.set_roi( roi )
		>>>  print(results)
		>>>  results.plot()
		'''
		try:
			roi  = RegionOfInterest(roi)
		except:
			try:
				roi  = RegionOfInterest(roi.value)
			except:
				raise ValueError('Input "roi" must be a viable input for power1d.roi.RegionOfInterest')
		assert isinstance(roi, RegionOfInterest), 'Input to set_roi must be an instance of power1d.RegionOfInterest.'
		assert roi.Q == self.Q, 'ROI and data continuum sizes must be the same. ROI size = %d, data size = %d' %(roi.Q, self.Q)
		self.roi   = roi
		self._calculate_probabilities()
		
		
	def sf(self, u):
		'''
		Survival function.
		
		The probability that the "effect" distribution exceeds an arbitrary
		threshold *u*.
		
		Arguments:
		
		u ------ threshold (float)
		
		Example:
		
		>>>  print(  results.sf( 3.0 )  )
		>>>
		>>>  u = [0, 1, 2, 3, 4, 5]
		>>>  p = [results.sf(uu) for uu in u]
		>>>  print( p )
		'''
		return (self.Z1.max(axis=1) > u).mean()






