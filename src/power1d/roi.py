'''
Regions of interest (ROIs)

ROIs define the continuum scope of the null hypothesis.

Example:

.. plot::
	:include-source:

	import numpy as np
	import matplotlib.pyplot as plt
	import power1d

	
	# create data sample models:
	J        = 8
	Q        = 101
	baseline = power1d.geom.Null( Q=Q )
	signal0  = power1d.geom.Null( Q=Q )
	signal1  = power1d.geom.GaussianPulse( Q=Q, q=40, fwhm=15, amp=3.5 )
	noise    = power1d.noise.Gaussian( J=J, Q=Q, mu=0, sigma=0.3 )
	model0   = power1d.models.DataSample(baseline, signal0, noise, J=J)
	model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)


	# assemble into experiment models:
	emodel0  = power1d.models.Experiment( [model0, model0], fn=power1d.stats.t_2sample )
	emodel1  = power1d.models.Experiment( [model0, model1], fn=power1d.stats.t_2sample )


	# simulate the experiments
	sim      = power1d.models.ExperimentSimulator(emodel0, emodel1)
	results  = sim.simulate(iterations=1000, progress_bar=True)


	# create ROI object and apply to the results:
	Q        = 101
	x        = np.array( [False] * Q )
	x[60:80] = True
	roi      = power1d.roi.RegionOfInterest(x)
	results.set_roi( roi )


	# plot:
	plt.close('all')
	results.plot()


.. note:: Since the ROI object limits the scope of the null hypothesis, \
power results pertain only to the continuum region(s) inside the ROI. \
In this example we see that the omnibus power is close to alpha because \
the ROI contains no signal. Simulating for a larger number of iterations \
will yield more precise convergence to alpha.
'''

# Copyright (C) 2023  Todd Pataky


import numpy as np
from scipy.ndimage import label as scipy_label
from . _base import _Continuum1D




class RegionOfInterest(_Continuum1D):
	'''
	Region of interest (ROI).
	
	Example:
	
	.. plot::
		:include-source:
	
		import numpy as np
		import matplotlib.pyplot as plt
		import power1d
		
		Q        = 101
		x        = np.array( [False] * Q )
		x[40:60] = True
		roi      = power1d.roi.RegionOfInterest(x)

		plt.close('all')
		roi.plot()
	'''
	
	def __init__(self, x):
		try:
			x       = np.array(x)
		except:
			raise ValueError('Input to RegionOfInterest must be a list or a numpy array')
		assert x.ndim==1, 'Input to RegionOfInterest must be a one-dimensional array'
		assert issubclass(x.dtype.type, np.bool_), 'Input to RegionOfInterest must only True and False values'
		assert x.sum()>0, 'Input to RegionOfInterest must contain at least one True value'
		Q       = x.size
		self.params = dict(x=x)
		super().__init__(Q)

	def _build(self):
		self.value  = self.params['x']
		
	def _get_labels(self):
		return scipy_label(self.value)

	def plot(self, ax=None, facecolor='b', edgecolor='b', alpha=0.25, q=None):
		from . _plot import DataPlotter
		plotter = DataPlotter(ax)
		plotter.plot_roi( q, self, facecolor, edgecolor, alpha )



