"""Fast checks that the library reproduces the exact numbers the papers quote.

Every quantity here is an exact spectral gap (1 minus the SLEM of an assembled transition matrix), so the
tolerances are tight.  The whole file runs in well under a minute on a laptop.
"""

import numpy as np
import pytest

from fibresampler.basis import RayView, free_points, hold_fixed_basis, int_det, recommend_basis, with_basis
from fibresampler.problems import P1, p1_poisson_theta, table_2x3
from fibresampler.reflective import reflective_transition_matrix
from fibresampler.spectral import log_weights, normalised_pi, slem, sr_transition_matrix
from fibresampler.synthetic import BandProblem


def plb_view(prob):
    return RayView(prob.states, prob.U, index=prob.index, plb_info=prob.plb_info)


def gap_of(view, lw):
    return slem(sr_transition_matrix(view, lw), normalised_pi(lw))[1]


def test_p1_fibre_and_uniform_gap():
    prob = P1()
    assert prob.fibre_size == 55
    assert gap_of(plb_view(prob), log_weights(prob, "uniform")) == pytest.approx(0.119828, abs=2e-6)


def test_loading_obstruction_and_hold_fixed_escape():
    """Paper II: the PLB gap is Theta(mu) in the mean mu of the bottleneck cell; recombining the basis so that one
    move set holds that cell fixed removes the obstruction (gap 0.1197 at mu = 1e-6, flat in mu)."""
    prob = P1()
    lw = log_weights(prob, "poisson", p1_poisson_theta(1e-6))
    V, n_slice = hold_fixed_basis(prob.U, [4])
    assert abs(int_det(V)) == 1
    assert n_slice == 3
    assert np.all((prob.U @ V)[4, :n_slice] == 0)
    view = plb_view(prob)
    assert gap_of(view, lw) < 2e-7
    assert gap_of(with_basis(view, V), lw) == pytest.approx(0.1197, abs=2e-4)


def test_selector_moves_on_a_thin_band_and_keeps_the_plb_on_a_round_fibre():
    """Paper I: on the band (w=2, C=24) the selected basis lifts the exact uniform-target gap from 0.0325 to 0.3341;
    on P1 the selector keeps the PLB."""
    band = BandProblem(2, 24)
    lw = np.zeros(band.fibre_size)
    view = RayView(band.states, band.U, index=band.index, plb_info=band.plb_info)
    V, info = recommend_basis(free_points(band), normalised_pi(lw))
    assert info["chosen"] != "identity"
    assert abs(int_det(V)) == 1
    assert gap_of(view, lw) == pytest.approx(0.0325, abs=1e-4)
    assert gap_of(with_basis(view, V), lw) == pytest.approx(0.3341, abs=1e-4)

    prob = P1()
    _, info_p1 = recommend_basis(free_points(prob), normalised_pi(log_weights(prob, "uniform")))
    assert info_p1["chosen"] == "identity"


@pytest.mark.parametrize("target", ["uniform", "poisson"])
def test_reflective_sampler_is_exactly_reversible(target):
    """The exact transition matrix of the reflective lattice sampler is stochastic and reversible."""
    prob = table_2x3()
    theta = np.array([0.7, 1.3, 2.0, 0.9, 1.1, 0.5])
    Q, _ = reflective_transition_matrix(prob, target=target, theta=theta, bmax=2)
    pi = normalised_pi(log_weights(prob, target, theta))
    assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-12)
    flow = pi[:, None] * Q
    assert np.abs(flow - flow.T).max() < 1e-12


def test_tempering_ladder_gap_and_detailed_balance():
    """Paper II, Task V: the three-rung-or-longer tempered chain on P1 at mu = 1e-3 with L = 10 has gap
    0.004587612051 and satisfies detailed balance to machine precision."""
    from taskV_tempering import gap_of_tempered

    gap, db_violation, n_rungs = gap_of_tempered(1e-3, 10)
    assert n_rungs == 11
    assert gap == pytest.approx(0.004587612051, rel=1e-9)
    assert db_violation < 1e-12
