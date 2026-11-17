'''
Datasets.

Currently only one dataset is available:

"weather"
'''

# Copyright (C) 2023  Todd Pataky


import os
import numpy as np


def weather():
	'''
	Canadian weather dataset from the R package "fda" (version 6.1.9)
	by James Ramsay, Giles Hooker and Spencer Graves, distributed under
	the license "GPL (>= 2)":
	https://cran.r-project.org/package=fda

	The data have been converted from "CanadianWeather.rda" to
	numpy format and saved in
	./power1d/data/weather/CanadianWeather.npz
	
	This is a convenience function for loading those data.
	
	Arguments:
	
	(None)
	
	Outputs:
	
	A dictionary containing the following keys:
	
	- "Atlantic"
	- "Pacific"
	- "Continental"
	- "Arctic"
	
	Each item is a (J x 365) array where J is the number of
	stations per region.
	
	Example:
	
	.. plot::
		:include-source:

		import matplotlib.pyplot as plt
		import power1d

		data = power1d.data.weather()   #load data dictionary
		y    = data['Continental']   #extract one region
		plt.plot(y.T, color="k")
	'''
	dir0     = os.path.dirname(__file__)
	fname    = os.path.join(dir0, 'data', 'weather', 'CanadianWeather.npz')
	with np.load(fname) as D:
		y      = D['dailyAv'][:,:,0].T
		region = D['region']
	z        = {name: y[region==name]  for name in ('Atlantic', 'Pacific', 'Continental', 'Arctic')}
	return z
