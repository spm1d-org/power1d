
import os
import numpy as np




def weather():
	'''
	Canadian weather dataset from the R package "fda" (version 6.1.9)
	by James Ramsay, Giles Hooker and Spencer Graves, distributed under
	the license "GPL (>= 2)":
	https://cran.r-project.org/package=fda
	(see the `./examples/data` directory).

	The dataset is described in:
	Ramsay JO, Silverman BW (2005). Functional Data Analysis
	(Second Edition), Springer, New York.
	Chapter 13: "Modelling functional responses with
	multivariate covariates"
	
	Data subsets include:

	- 'Atlantic'
	- 'Pacific'
	- 'Continental'
	- 'Arctic'
	
	
	:Example use:
	
	>>> data = weather()
	>>> y = data['Atlantic']  # (15 x 365) numpy array
	>>> import matplotlib.pyplot as plt
	>>> plt.plot(y.T)
	'''
	dir0     = os.path.dirname( __file__ )
	fname    = os.path.join(dir0, 'data', 'CanadianWeather.npz')
	with np.load(fname) as Z:
		Y      = Z['dailyAv'][:,:,0].T
		region = Z['region']
	D        = {name: Y[region==name]  for name in ('Atlantic', 'Pacific', 'Continental', 'Arctic')}
	return D




#(0) Load data:
data = weather() #dictionay containing geographical locations



#(1) Save as dictionary in an NPZ file:
dir0     = os.path.dirname( __file__ )
fname    = os.path.join(dir0, 'data', 'daily.npz')
np.savez(fname, **data)





