'''
Serialisable form: obj._to_dict() / power1d._base._Power1DObject._from_dict(d).
'''
import json

import numpy as np
import pytest
import power1d
from power1d._base import _Power1DObject

Q = 101


GEOMS = [
	power1d.geom.Constant(Q, amp=2.5),
	power1d.geom.Exponential(Q, x0=0.2, x1=3, rate=5),
	power1d.geom.ExponentialSaw(Q, x0=0, x1=2, rate=3, cutoff=60),
	power1d.geom.GaussianPulse(Q, q=40, fwhm=15, amp=1.2),
	power1d.geom.GaussianPulse(Q, q=40, sigma=6, amp=1.2),
	power1d.geom.Linear(Q, x0=0, x1=2),
	power1d.geom.Linear(Q, x0=1, slope=0.1),
	power1d.geom.Null(Q),
	power1d.geom.SawPulse(Q, q0=20, q1=60, x0=0, x1=3),
	power1d.geom.SawTooth(Q, q0=10, q1=30, x0=0, x1=1, dq=5),
	power1d.geom.Sigmoid(Q, q0=20, q1=60, x0=0, x1=4),
	power1d.geom.Sinusoid(Q, q0=5, amp=2, hz=3),
	power1d.geom.SquarePulse(Q, q0=20, q1=60, x0=0, x1=3),
	power1d.geom.SquareTooth(Q, q0=10, q1=25, x0=0, x1=1, dq=5),
	power1d.geom.TrianglePulse(Q, q0=20, q1=60, x0=0, x1=3),
	power1d.geom.TriangleTooth(Q, q0=10, q1=30, x0=0, x1=1, dq=20),
	power1d.geom.GaussianPulse(Q, q=50, fwhm=40, amp=5) - power1d.geom.Constant(Q, amp=2),
]


def _noises():
	g   = power1d.noise.Gaussian(J=8, Q=Q, mu=0, sigma=1)
	sg  = power1d.noise.SmoothGaussian(J=8, Q=Q, mu=0, sigma=1, fwhm=20)
	return [
		power1d.noise.ConstantGaussian(J=8, Q=Q, mu=1, sigma=2),
		power1d.noise.ConstantUniform(J=8, Q=Q, x0=-1, x1=1),
		g,
		power1d.noise.Skewed(J=8, Q=Q, mu=0, sigma=1, alpha=3),
		sg,
		power1d.noise.SmoothSkewed(J=8, Q=Q, mu=0, sigma=1, fwhm=15, alpha=2),
		power1d.noise.Uniform(J=8, Q=Q, x0=0, x1=2),
		power1d.noise.Additive(g, sg),
		power1d.noise.Mixture(power1d.noise.Gaussian(J=3, Q=Q), power1d.noise.Gaussian(J=5, Q=Q, sigma=2)),
		power1d.noise.Scaled(g, np.linspace(0, 1, Q)),
		power1d.noise.SignalDependent(g, power1d.geom.GaussianPulse(Q, q=60, fwhm=15, amp=3)),
	]


def _json_roundtrip(d):
	'''The dictionary must survive a plain-JSON round trip (arrays as lists).'''
	def enc(o):
		return o.tolist() if isinstance(o, np.ndarray) else o
	return json.loads(json.dumps(d, default=enc))


@pytest.mark.parametrize('g', GEOMS, ids=lambda g: type(g).__name__)
def test_geometry_round_trip(g):
	d = g._to_dict()
	b = _Power1DObject._from_dict(d)
	assert type(b) is type(g)
	assert np.allclose(b.value, g.value)
	assert np.allclose(_Power1DObject._from_dict(_json_roundtrip(d)).value, g.value)


@pytest.mark.parametrize('n', _noises(), ids=lambda n: type(n).__name__)
def test_noise_round_trip(n):
	d = n._to_dict()
	b = _Power1DObject._from_dict(d)
	assert type(b) is type(n)
	assert b.value.shape == n.value.shape
	assert b._to_dict().keys() == d.keys()
	# same distribution parameters: statistics of many draws agree with the original model
	np.random.seed(0)
	x0 = np.array([n._get_new_random_value().copy() for _ in range(200)])
	np.random.seed(0)
	x1 = np.array([b._get_new_random_value().copy() for _ in range(200)])
	assert np.allclose(x0, x1)


def test_custom_functions_are_refused():
	g = power1d.noise.Gaussian(J=8, Q=Q)
	s = power1d.geom.GaussianPulse(Q, q=60, fwhm=15, amp=3)
	with pytest.raises(ValueError):
		power1d.noise.SignalDependent(g, s, fn=lambda n, s: n * s)._to_dict()
	m = power1d.models.DataSample(power1d.geom.Null(Q), power1d.geom.Null(Q), g, J=8)
	with pytest.raises(ValueError):
		power1d.models.Experiment([m], fn=lambda y: y.mean(axis=0))._to_dict()


def _simulator(J=8):
	baseline = power1d.geom.Null(Q=Q)
	signal1  = power1d.geom.GaussianPulse(Q=Q, q=40, fwhm=15, amp=1.2)
	noise    = power1d.noise.SmoothGaussian(J=J, Q=Q, mu=0, sigma=1.0, fwhm=20)
	model0   = power1d.models.DataSample(baseline, power1d.geom.Null(Q=Q), noise, J=J)
	model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)
	e0 = power1d.models.Experiment([model0, model0], fn=power1d.stats.t_2sample)
	e1 = power1d.models.Experiment([model0, model1], fn=power1d.stats.t_2sample)
	return power1d.models.ExperimentSimulator(e0, e1)


def test_models_round_trip_simulate_identically():
	sim = _simulator()
	b   = _Power1DObject._from_dict(_json_roundtrip(sim._to_dict()))
	assert type(b) is power1d.models.ExperimentSimulator
	np.random.seed(3)
	r0  = sim.simulate(60, progress_bar=False)
	np.random.seed(3)
	r1  = b.simulate(60, progress_bar=False)
	assert np.allclose(r0.Z0, r1.Z0) and np.allclose(r0.Z1, r1.Z1)


@pytest.mark.parametrize('fn', ['t_1sample_fn', 't_2sample_fn', 'f_anova1_fn'])
def test_fast_statistic_functions_round_trip(fn):
	J  = 8
	m  = power1d.models.DataSample(power1d.geom.Null(Q), power1d.geom.Null(Q), power1d.noise.Gaussian(J=J, Q=Q), J=J)
	f  = dict(t_1sample_fn=lambda: power1d.stats.t_1sample_fn(J), t_2sample_fn=lambda: power1d.stats.t_2sample_fn(J, J),
	          f_anova1_fn=lambda: power1d.stats.f_anova1_fn(J, J, J))[fn]()
	n  = dict(t_1sample_fn=1, t_2sample_fn=2, f_anova1_fn=3)[fn]
	e  = power1d.models.Experiment([m] * n, fn=f)
	b  = _Power1DObject._from_dict(_json_roundtrip(e._to_dict()))
	y  = [np.random.randn(J, Q) for _ in range(n)]
	assert np.allclose(b.fn(*y), e.fn(*y))


def test_regressor_model_round_trip():
	x  = np.linspace(0, 1, 10)
	m  = power1d.models.DataSample(power1d.geom.Null(Q), power1d.geom.GaussianPulse(Q, q=50, fwhm=20, amp=1),
	                              power1d.noise.Gaussian(J=10, Q=Q), J=10, regressor=x)
	e  = power1d.models.Experiment(m, fn=power1d.stats.t_regress_fn(x))
	b  = _Power1DObject._from_dict(e._to_dict())
	assert np.allclose(b.data_models[0].regressor, x)


def test_results_round_trip_and_settings():
	np.random.seed(0)
	r = _simulator().simulate(100, progress_bar=False)
	roi = np.zeros(Q, dtype=bool)
	roi[20:80] = True
	r.set_roi(roi)
	r.set_coi((40, 3), (60, 5))
	r.set_poi(40, 50)
	r.set_alpha(0.01)
	d = r._to_dict()
	for b in (_Power1DObject._from_dict(d), _Power1DObject._from_dict(_json_roundtrip(d))):
		# the rebuilt experiment models draw new random values, so compare the results themselves
		assert np.array_equal(b.Z0, r.Z0) and np.array_equal(b.Z1, r.Z1)
		assert (b.alpha, b.k, b.coir) == (r.alpha, r.k, r.coir)
		assert np.array_equal(b.roi.value, r.roi.value)
		assert b.zstar == pytest.approx(r.zstar)
		assert b.p_reject0 == pytest.approx(r.p_reject0) and b.p_reject1 == pytest.approx(r.p_reject1)
		assert np.allclose(b.p1d_coi1, r.p1d_coi1) and np.allclose(b.p1d_poi1, r.p1d_poi1)
		assert np.allclose(b.p_coi1, r.p_coi1) and np.allclose(b.p_poi1, r.p_poi1)
		assert type(b.model0) is power1d.models.Experiment
