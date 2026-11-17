'''
Theoretical continuum-level probabilities for central t fields
and for noncentral t and F fields.

These calculations come from the random field theory (RFT)
literature and in particular from the three references listed
below. The main RFT calculations come from Hasofer (1978) and
Friston et al. (1994) and the non-central distribution
calculations come from Hayasaka et al. (2007) and
Joyce & Hayasaka (2012).

REFERENCES

| Friston KJ, Holmes A, Poline JB, Price CJ, Frith CD (1996).
  Detecting Activations in PET and fMRI: Levels of Inference
  and Power. NeuroImage 4(3): 223-235.
| http://dx.doi.org/10.1006/nimg.1996.0074


| Friston KJ, Worsley KJ, Frackowiak RSJ, Mazziotta JC, Evans AC
  (1994). Assessing the significance of focal activations using
  their spatial extent. **Human Brain Mapp** 1: 210-220.
| http://onlinelibrary.wiley.com/doi/10.1002/hbm.460010306/full
| SPM12 software:
| http://www.fil.ion.ucl.ac.uk/spm/software/spm12/

| Hasofer AM (1978) Upcrossings of random fields.
  **Suppl Adv Appl Prob** 10:14-21.
| http://www.jstor.org/stable/1427002

| Hayasaka S, Peiffer AM, Hugenschmidt CE, Laurienti PJ (2007).
  Power and sample size calculation for neuroimaging studies by
  non-central random field theory. **NeuroImage** 37(3), 721-730.
| http://dx.doi.org/10.1016/j.neuroimage.2007.06.009

| Joyce KE, Hayasaka S (2012). Development of PowerMap: a Software Package for
  Statistical Power Calculation in Neuroimaging Studies **Neuroinform** 10: 351.
| http://dx.doi:10.1007/s12021-012-9152-3
| PowerMap software:
| https://sourceforge.net/projects/powermap/
'''

# Copyright (C) 2023  Todd Pataky



from math import pi,log,sqrt,exp
import numpy as np
from scipy import stats,optimize
from scipy.special import gammaln,gamma
from . import geom

# CONSTANTS:
FOUR_LOG2   = 4*log(2)
SQRT_4LOG2  = sqrt(4*log(2))
SQRT_2      = sqrt(2)
TWO_PI      = 2*pi
eps         = np.finfo(float).eps





def exp_ncx(df, delta, r):
	'''
	Non-central chi-square moments around zero

	Adapted from "pm_Exp_ncX.m" in "PowerMap" (Joyce & Hayasaka 2012)

	Arguments:

	*df* ---- degrees of freedom (float)

	*delta* ---- non-centrality parameter (float)

	*r* ---- power (positive float)
	'''
	tol     = eps**7.0/8
	isum    = 0
	iisum   = 1
	j       = 0
	while abs(iisum) > tol:
		gam    = 1 / gamma(j+1)
		d      = exp( gammaln(r+j+0.5*df)  - gammaln(j + 0.5*df)  )  * (0.5*delta)**j
		iisum  = gam * d
		isum  += iisum
		j     += 1
	mr      = 2**r  * exp(-0.5*delta) * isum
	return mr



def ec_density_ncf(z, v, delta):
	'''
	Euler characteristic density for non-central F fields.

	Adapted from "pm_ECncF.m" in "PowerMap" (Joyce & Hayasaka 2012)

	Arguments:

	*z* ---- height (float)

	*v* ---- degrees of freedom (float)

	*delta* ---- non-centrality parameter (float)
	'''
	v0,v1    = v
	a        = FOUR_LOG2
	b        = TWO_PI
	c        = v0 * z / v1
	d        = 1 + c
	ec0      = stats.ncf.sf(z, v0, v1, delta)
	ec1      = a**0.5 * b**-0.5 * 2 * d * c**0.5 * exp_ncx(v0+v1, delta, -0.5) * (v1/v0) * stats.ncf.pdf(z,v0,v1,delta)
	return np.maximum([ec0, ec1], eps)


def ec_density_nct(z, v, delta):
	'''
	Euler characteristic density for non-central t fields.

	Adapted from "pm_ECncT.m" in "PowerMap" (Joyce & Hayasaka 2012)

	Arguments:

	*z* ---- height (float)

	*v* ---- degrees of freedom (float)

	*delta* ---- non-centrality parameter (float)
	'''
	a    = FOUR_LOG2
	b    = TWO_PI
	c    = z**2 / v
	d    = 1 + c
	ec0  = stats.nct.sf(z, v, delta)
	ec1  = a**0.5 * b**-0.5 * v**0.5 * d * exp_ncx(v+1, delta**2, -0.5)  * stats.nct.pdf(z, v, delta)
	return np.maximum([ec0, ec1], eps)


# Translated from SPM12 spm_ECdensity.m:
# Copyright (C) 1999-2013 Wellcome Trust Centre for Neuroimaging
# Karl Friston
def ec_density_f(z, v, delta=None):
	'''
	Euler characteristic density for central F fields.

	Adapted from "spm_ECdensity.m" in "SPM12" (Friston et al. 1994)

	Arguments:

	*z* ---- height (float)

	*v* ---- degrees of freedom (float, float)

	*delta* ---- (not used by this function)
	
	Reference:  Worsley KJ et al. (1996) Hum Brain Mapp 4:58-73
	Reference:  Worsley KJ et al. (2004) [Eqn.2 and Table 2]
	'''
	k,v  = map(float, v)
	k    = max(k, 1.0)        #stats.f.cdf will return nan if k is less than 1
	a    = FOUR_LOG2/TWO_PI
	b    = gammaln(v/2) + gammaln(k/2)
	ec0  = stats.f.sf(z, k, v)
	ec1  = a**0.5 * np.exp(gammaln((v+k-1)/2)-b)*2**0.5 *(k*z/v)**(0.5*(k-1))*(1+k*z/v)**(-0.5*(v+k-2))
	return np.maximum([ec0, ec1], eps)


# Translated from SPM12 spm_ECdensity.m:
# Copyright (C) 1999-2013 Wellcome Trust Centre for Neuroimaging
# Karl Friston
def ec_density_t(z, v, delta=None):
	'''
	Euler characteristic density for central t fields.

	Adapted from "spm_ECdensity.m" in "SPM12" (Friston et al. 1994)

	Arguments:

	*z* ---- height (float)

	*v* ---- degrees of freedom (float)

	*delta* ---- (not used by this function)
	
	Reference:  Worsley KJ et al. (1996) Hum Brain Mapp 4:58-73
	Reference:  Worsley KJ et al. (2004) [Eqn.2 and Table 2]
	'''
	v    = float(v)
	a    = FOUR_LOG2
	b    = np.exp(  (gammaln((v+1)/2) - gammaln(v/2))  )
	c    = (1+z**2/v)**((1-v)/2)
	ec0  = stats.t.sf(z,v)
	ec1  = a**0.5 / TWO_PI * c
	return np.maximum([ec0, ec1], eps)


ec_fn_dict = {
	'NCF' : ec_density_ncf,
	'NCT' : ec_density_nct,
	'F'   : ec_density_f,
	'T'   : ec_density_t
}



def _rft_sf(STAT, u, df, Q, fwhm, delta=None):
	'''
	Random field theory (RFT) survival function
	
	References:
	Hasofer (1978) and Friston et al. (1994)
	
	Also:
	Worsley KJ et al. (1996) Hum Brain Mapp 4:58-73
	Worsley KJ et al. (2004) [Eqn.2 and Table 2]
	'''
	_stats   = ['F','T','NCT','NCF']
	assert STAT in _stats, 'STAT must be one of: %s' %_stats
	df       = map(float, df) if ('F' in STAT) else float(df)
	resels   = np.asarray( [1.0, float(Q-1)/fwhm] )   #resel counts
	ec_fn    = ec_fn_dict[STAT]                       #Euler characteristic density function
	EC       = ec_fn( float(u), df, delta)            #Euler characteristic
	Ec       = sum( resels * EC )                     #expected number of maxima
	p        = stats.poisson.sf(0, Ec + eps)          #Poisson clumping heuristic
	return p	




def ncf_sf(u, df, Q, fwhm, delta):
	'''
	Survival function for noncentral f fields.

	Adapted from "pm_P_ncT.m" in "PowerMap" (Joyce & Hayasaka 2012)

	Arguments:

	*u* ---- height (float)

	*df* ---- degrees of freedom (float)
	
	*Q* ---- continuum size (integer)
	
	*fwhm* ---- continuum smoothness (positive float);  full-width-at-half-maximum

	*delta* ---- noncentrality parameter
	'''
	return _rft_sf('NCF', u, df, Q, fwhm, delta)


def nct_sf(u, df, Q, fwhm, delta):
	'''
	Survival function for noncentral t fields.

	Adapted from "pm_P_ncT.m" in "PowerMap" (Joyce & Hayasaka 2012)

	Arguments:

	*u* ---- height (float)

	*df* ---- degrees of freedom (float)
	
	*Q* ---- continuum size (integer)
	
	*fwhm* ---- continuum smoothness (positive float);  full-width-at-half-maximum

	*delta* ---- noncentrality parameter
	'''
	return _rft_sf('NCT', u, df, Q, fwhm, delta)


def f_sf(u, df, Q, fwhm):
	'''
	Survival function for central F fields.

	Arguments:

	*u* ---- height (float)

	*df* ---- degrees of freedom (float)
	
	*Q* ---- continuum size (integer)
	
	*fwhm* ---- continuum smoothness (positive float);  full-width-at-half-maximum
	'''
	return _rft_sf('F', u, df, Q, fwhm)



def t_sf(u, df, Q, fwhm):
	'''
	Survival function for central t fields.

	Arguments:

	*u* ---- height (float)

	*df* ---- degrees of freedom (float)
	
	*Q* ---- continuum size (integer)
	
	*fwhm* ---- continuum smoothness (positive float);  full-width-at-half-maximum
	'''
	return _rft_sf('T', u, df, Q, fwhm)
	
	

def f_isf(alpha, df, Q, fwhm):
	'''
	Inverse survival function for central f fields
	
	Arguments:

	*alpha* ---- Type I error rate (float between 0 and 1)

	*df* ---- degrees of freedom (float, float)
	
	*Q* ---- continuum size (integer)
	
	*fwhm* ---- continuum smoothness (positive float);  full-width-at-half-maximum
	'''
	z0    = stats.f.isf(alpha, df[0], df[1])  #approximate threshold
	fn    = lambda x : (f_sf(x, df, Q, fwhm) - alpha)**2
	zstar = optimize.fmin(fn, z0, xtol=1e-9, disp=0)[0]
	return zstar


def t_isf(alpha, df, Q, fwhm):
	'''
	Inverse survival function for central t fields
	
	Arguments:

	*alpha* ---- Type I error rate (float between 0 and 1)

	*df* ---- degrees of freedom (float)
	
	*Q* ---- continuum size (integer)
	
	*fwhm* ---- continuum smoothness (positive float);  full-width-at-half-maximum
	'''
	z0    = stats.t.isf(alpha, df)  #approximate threshold
	fn    = lambda x : (t_sf(x, df, Q, fwhm) - alpha)**2
	zstar = optimize.fmin(fn, z0, xtol=1e-9, disp=0)[0]
	return zstar
	
	

def power_Friston1994(alpha, df, Q, W0, W1, sigma):
	'''
	Continuum-level power calculation using the inflated variance method
	(Friston et al. 1994)

	Arguments:

	*alpha* ------ Type I error rate  (float)
	
	*df* ------ degrees of freedom  (float)
	
	*Q* ------ continuum size  (int)
	
	*W0* ------ continuum smoothness under the null hypothesis  (positive float)
	
	*W1* ------ continuum smoothness under the alternative hypothesis  (float)
	
	*sigma* ------ effect size  (noise amplitude under the alternative hypothesis)  (positive float)
	
	Reference:
	
	| Friston KJ, Worsley KJ, Frackowiak RSJ, Mazziotta JC, Evans AC
	  (1994). Assessing the significance of focal activations using
	  their spatial extent. **Human Brain Mapp** 1: 210-220.
	| http://onlinelibrary.wiley.com/doi/10.1002/hbm.460010306/full
	'''
	f      = float(W1) / W0                  #ratio of signal-to-noise smoothness
	s2     = sigma**2                        #variance
	u      = t_isf(alpha, df, Q, W0)         #critical threshold
	ustar  = u * ( 1 + s2 )**(-0.5)          #threshold for the alternative hypothesis
	Wstar  = W0 * ( (1+s2) / ( 1 + s2/(1+f**2)) )**0.5  #smoothness for the alternative
	p      = t_sf(ustar, df, Q, Wstar)       #power
	return p


def power_Hayasaka2007(alpha, df, Q, fwhm, effect):
	'''
	Continuum-level power calculation using the non-central t method
	(Hayasaka et al. 2007)
	
	Arguments:
	
	*alpha* ------ Type I error rate  (float)
	
	*df* ------ degrees of freedom  (float)
	
	*Q* ------ continuum size  (int)
	
	*fwhm* ------ continuum smoothness  (positive float)
	
	*effect* ------ effect size  (random field shift as: mu/sigma) (float)
	
	Reference:
	
	| Hayasaka S, Peiffer AM, Hugenschmidt CE, Laurienti PJ (2007).
	  Power and sample size calculation for neuroimaging studies by
	  non-central random field theory. **NeuroImage** 37(3), 721-730.
	| http://dx.doi.org/10.1016/j.neuroimage.2007.06.009
	'''
	delta  = effect * (df+1)**0.5   #non-centrality parameter
	u      = t_isf(alpha, df, Q, fwhm)
	p      = nct_sf(u, df, Q, fwhm, delta)
	return p




def power(alpha, df, Q, fwhm, effect, method='nct'):
	'''
	Continuum-level theoretical power calculations for one-sample t tests.
	
	This is a convenience function for generating comparable results for
	the "iv" and "nct" methods (see below). That is, the two methods define
	effects slightly differently, but this function ensures that the effect
	is defined approximately the same for both methods, thus making the
	two methods directly comparable.
	
	
	*alpha* ------ Type I error rate  (float)
	
	*df* ------ degrees of freedom  (float)
	
	*Q* ------ continuum size  (int)
	
	*fwhm* ------ continuum smoothness  (positive float)
	
	*effect* ------ effect size  (float)
	
	*method* ------ one of "iv" or "nct"
	
	Methods:
	
	*iv* ------ inflated variance method (Friston et al. 1994)
	
	*nct* ------ noncentral t method (Hayasaka et al. 2007)
	'''
	_methods = ['iv', 'nct']
	assert method in _methods, '"method" must be one of: %s' %_methods
	if method == 'iv':
		d    = effect * (df+1)**0.5
		p    = power_Friston1994(alpha, df, Q, fwhm, fwhm, d)
	if method == 'nct':
		p    = power_Hayasaka2007(alpha, df, Q, fwhm, effect)
	return p
	# method='iv'  #inflated_variance
	# method='ncrft'  #noncentral_rft
	pass


