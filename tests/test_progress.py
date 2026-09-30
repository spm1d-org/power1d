import numpy as np
import power1d


def _simulator():
    J, Q = 8, 101
    baseline = power1d.geom.Null(Q=Q)
    signal0  = power1d.geom.Null(Q=Q)
    signal1  = power1d.geom.GaussianPulse(Q=Q, q=40, fwhm=15, amp=1.2)
    noise    = power1d.noise.Gaussian(J=J, Q=Q, mu=0, sigma=1.0)
    model0   = power1d.models.DataSample(baseline, signal0, noise, J=J)
    model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)
    emodel0  = power1d.models.Experiment([model0, model0], fn=power1d.stats.t_2sample)
    emodel1  = power1d.models.Experiment([model0, model1], fn=power1d.stats.t_2sample)
    return power1d.models.ExperimentSimulator(emodel0, emodel1), emodel1


def test_simulator_progress_callback():
    sim, _ = _simulator()
    calls  = []
    sim.simulate(60, progress_bar=False, progress=lambda i, n: calls.append((i, n)))
    assert calls == [(i, 60) for i in range(1, 61)]


def test_experiment_progress_callback():
    _, emodel = _simulator()
    calls  = []
    emodel.simulate(20, progress_bar=False, progress=lambda i, n: calls.append((i, n)))
    assert calls[-1] == (20, 20) and len(calls) == 20


def test_results_unchanged_by_callback():
    sim, _ = _simulator()
    np.random.seed(0)
    r0 = sim.simulate(60, progress_bar=False)
    np.random.seed(0)
    r1 = sim.simulate(60, progress_bar=False, progress=lambda i, n: None)
    assert np.array_equal(r0.Z0, r1.Z0) and np.array_equal(r0.Z1, r1.Z1)
