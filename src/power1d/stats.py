'''
A module containing test statistic functions for simple
experiment designs.

The following functions are available:

- t_1sample_fn  ------ one sample t test
- t_2sample_fn ------ two sample t test
- t_regress_fn ------ linear regression
- f_anova1_fn ------ one-way ANOVA

All functions accept two-dimensional numpy arrays as
the dependent variable input argument(s). The arrays
must have shape (J,Q) where:

- J ------ sample size
- Q ------ continuum size

All functions return a test statistic continuum as a
one-dimensional numpy array of size Q.


Slightly more efficient versions of the functions above
are also available:

- t_1sample_fn
- t_2sample_fn
- t_regress_fn
- f_anova1_fn

The output from each of these functions is itself a function
whose input arguments are identical to the normal versions above.
However, the _fn versions store information like degrees of
freedom and matrix inversion results so they needn't be
re-computed.  This makes iterative simulation somewhat more
efficient.
'''

# Copyright (C) 2023  Todd Pataky



from math import sqrt
import numpy as np


eps           = np.finfo(float).eps



#------------------------------------
# One-sample t statistic
#------------------------------------
def t_1sample(y):
	'''
	t statistic for a one-sample test
	
	Arguments:
	
	y ------ (J x Q) data sample array
	
	Outputs:
	
	t continuum as a numpy array with shape = (Q,)
	
	
	Example:
	
	>>> import numpy as np
	>>> import power1d
	>>> J,Q   = 8, 101
	>>> y     = np.random.randn( J, Q )
	>>> t     = power1d.stats.t_1sample( y )
	'''
	return y.mean(axis=0) / (   y.std(ddof=1, axis=0) / sqrt(y.shape[0])   )

def t_1sample_fn(J):
	'''
	t statistic for a one-sample test
	
	Arguments:
	
	J ------ sample size
	
	Outputs:
	
	A function for computing the t statistic.
	
	Example:
	
	>>> import numpy as np
	>>> import power1d
	>>> J,Q   = 8, 101
	>>> y     = np.random.randn( J, Q )
	>>> fn    = power1d.stats.t_1sample_fn( J )
	>>> t     = fn( y )
	'''
	sqrtJ = sqrt( J )
	def fn(y):
		return y.mean(axis=0) / (   y.std(ddof=1, axis=0) / sqrtJ   )
	fn._factory = ('t_1sample_fn', (J,))   # recorded for serialisation
	return fn




#------------------------------------
# Two-sample t statistic
#------------------------------------

def t_2sample(yA, yB):
	'''
	t statistic for a two-sample test
	
	Arguments:
	
	yA ------ (J x Q) data sample array
	
	yB ------ (J x Q) data sample array
	
	Outputs:
	
	t continuum as a numpy array with shape = (Q,)
	
	
	Example:
	
	>>> import numpy as np
	>>> import power1d
	>>> J,Q   = 8, 101
	>>> yA    = np.random.randn( J, Q )
	>>> yB    = np.random.randn( J, Q )
	>>> t     = power1d.stats.t_2sample( yA, yB )
	'''
	JA,JB  = yA.shape[0], yB.shape[0]
	mA,mB  = yA.mean(axis=0), yB.mean(axis=0)
	sA,sB  = yA.std(ddof=1, axis=0), yB.std(ddof=1, axis=0)
	s      = (   (  (JA-1)*sA*sA + (JB-1)*sB*sB  )  /  ( JA+JB-2 )   )**0.5
	t      = (mB-mA) / s / (1.0/JA + 1.0/JB)**0.5
	return t

def t_2sample_fn(JA, JB):
	'''
	t statistic for a two-sample test
	
	Arguments:
	
	JA ------ sample size for group A
	
	JB ------ sample size for group B
	
	Outputs:
	
	A function for computing the t statistic.
	
	
	Example:
	
	>>> import numpy as np
	>>> import power1d
	>>> JA,JB = 8, 10
	>>> Q     = 101
	>>> yA    = np.random.randn( J, Q )
	>>> yB    = np.random.randn( J, Q )
	>>> fn    = power1d.stats.t_2sample_fn( JA, JB )
	>>> t     = fn( yA, yB )
	'''
	JA1,JB1    = JA-1, JB-1
	JAB2       = JA + JB - 2
	sqrt1JA1JB = (1.0/JA + 1.0/JB)**0.5
	def fn(yA, yB):
		mA,mB  = yA.mean(axis=0), yB.mean(axis=0)
		sA,sB  = yA.std(ddof=1, axis=0), yB.std(ddof=1, axis=0)
		s      = (   ( JA1*sA*sA + JB1*sB*sB )  /  JAB2   )**0.5
		t      = (mB-mA) / s / sqrt1JA1JB
		return t
	fn._factory = ('t_2sample_fn', (JA, JB))   # recorded for serialisation
	return fn



#------------------------------------
# Linear regression t statistic
#------------------------------------


def t_regress(y, x):
	'''
	t statistic for linear regression
	
	Arguments:
	
	y ------ (J x Q) data sample array
	
	x ------ (J,) array of independent variable values
	
	Outputs:
	
	t continuum as a numpy array with shape = (Q,)
	
	
	Example:
	
	>>> import numpy as np
	>>> import power1d
	>>> J,Q   = 8, 101
	>>> y     = np.random.randn( J, Q )
	>>> x     = np.random.randn( J )
	>>> t     = power1d.stats.t_regress( y, x )
	'''
	J      = y.shape[0]
	X      = np.ones(  (J,2) )
	X[:,0] = x
	c      = np.array( [1,0] )
	b      = np.linalg.pinv(X) @ y   # regression parameters
	eij    = y - X @ b               # residuals
	R      = eij.T @ eij             # residual sum of squares
	df     = J - 2                   # degrees of freedom
	s2     = np.diag(R) / df         # variance
	t      = np.array(c.T @ b)  /   np.sqrt(  s2 * (c.T @ (np.linalg.inv(X.T @ X) )@c)  )
	return t



def t_regress_fn(x):
	'''
	t statistic for linear regression
	
	Arguments:
	
	x ------ (J,) array of independent variable values
	
	Outputs:
	
	A function for computing the t statistic.
	
	
	Example:
	
	>>> import numpy as np
	>>> import power1d
	>>> J,Q   = 8, 101
	>>> y     = np.random.randn( J, Q )
	>>> x     = np.random.randn( J )
	>>> fn    = power1d.stats.t_regress_fn( x )
	>>> t     = fn( y )
	'''
	J      = x.size
	X      = np.ones( (J,2) )
	X[:,0] = x
	c      = np.array( [1,0] )
	Xi     = np.linalg.pinv(X)
	cXXc   = c.T @ (  np.linalg.inv(X.T @ X)  ) @ c
	df     = J - 2                    # degrees of freedom
	def fn(y):
		b      = Xi @ y               # regression parameters
		eij    = y - X@b              # residuals
		R      = eij.T @ eij          # residual sum of squares
		s2     = np.diag(R) / df      # variance
		t      = (c.T @ b)  /   np.sqrt( s2 * cXXc )
		return t
	fn._factory = ('t_regress_fn', (x,))   # recorded for serialisation
	return fn
	
	

#------------------------------------
# One-way ANOVA F statistic
#------------------------------------

def _anova1_design_matrices(nResponses):
	nTotal    = sum(nResponses)
	nGroups   = len(nResponses)
	X         = np.zeros( (nTotal,nGroups) )
	i0        = 0
	for i,n in enumerate(nResponses):
		X[i0:i0+n,i] = 1
		i0   += n
	X0        = np.ones((nTotal,1))  # reduced design matrix
	Xi,X0i    = np.linalg.pinv(X), np.linalg.pinv(X0)  #pseudo-inverses
	return X,X0,Xi,X0i


def f_anova1(*yy):
	'''
	F statistic for a one-way ANOVA
	
	Arguments:
	
	yy ------ an arbitrary number of (J x Q) data sample arrays
	
	Outputs:
	
	f continuum as a numpy array with shape = (Q,)
	
	
	Example:
	
	>>> import numpy as np
	>>> import power1d
	>>> Q     = 8, 101
	>>> yA    = np.random.randn( 8, Q )
	>>> yB    = np.random.randn( 5, Q )
	>>> yC    = np.random.randn( 12, Q )
	>>> f     = power1d.stats.f_anova1( yA, yB, yC )
	'''
	nGroups     = len(yy)
	nResponses  = [a.shape[0]  for a in yy]
	nTotal      = sum( nResponses )
	v0,v1       = nGroups-1, nTotal-nGroups  # degrees of freedom
	X,X0,Xi,X0i = _anova1_design_matrices( nResponses )
	# estimate parameters:
	y           = np.vstack(yy)
	b           = Xi @ y
	eij         = y - X @ b
	R           = eij.T @ eij
	# reduced design:
	b0          = X0i @ y
	eij0        = y - X0 @ b0
	R0          = eij0.T @ eij0
	# F statistic:
	f           = ( (np.diag(R0)-np.diag(R)) / v0)   /   ( np.diag(R+eps) / v1 )
	return f


def f_anova1_fn(*JJ):
	'''
	F statistic for a one-way ANOVA
	
	Arguments:
	
	JJ ------ an arbitrary number sample sizes
	
	Outputs:
	
	A function for computing the f statistic.
	
	
	Example:
	
	>>> import numpy as np
	>>> import power1d
	>>> JA    = 8
	>>> JB    = 12
	>>> JC    = 9
	>>> Q     = 101
	>>> yA    = np.random.randn( JA, Q )
	>>> yB    = np.random.randn( JB, Q )
	>>> yC    = np.random.randn( JC, Q )
	>>> fn    = power1d.stats.f_anova1_fn( JA, JB, JC )
	>>> f     = fn( yA, yB, yC )
	'''
	nGroups     = len(JJ)
	nTotal      = sum(JJ)
	v0,v1       = nGroups-1, nTotal-nGroups  # degrees of freedom
	X,X0,Xi,X0i = _anova1_design_matrices( JJ )
	def fn(*yy):
		y       = np.vstack(yy)
		# estimate parameters:
		y       = np.vstack(yy)
		b       = Xi @ y
		eij     = y - X @ b
		R           = eij.T @ eij
		# reduced design:
		b0      = X0i @ y
		eij0    = y - X0 @ b0
		R0      = eij0.T @ eij0
		# F statistic:
		f       = ( (np.diag(R0)-np.diag(R)) / v0)   /   ( np.diag(R+eps) / v1 )
		return f
	fn._factory = ('f_anova1_fn', JJ)   # recorded for serialisation
	return fn
