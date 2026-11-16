# Copyright (C) 2025 The Cooper Developers.
# Licensed under the MIT License.

import pytest
import torch

import cooper
from cooper.optim import ExtraNuPI, ExtraSGD

TARGET = torch.tensor([0.3, -1.0, 2.0])


def error_fn(param):
    """A 'constraint violation' that depends on the current dual iterate."""
    return TARGET - 0.5 * param + 0.1 * param**2


def run_extragradient(optimizer, param, num_steps):
    """Run extragradient, setting p.grad to the error at each half-step.
    Returns the iterates after each actual step and the errors used."""
    iterates, errors = [], []
    for _ in range(num_steps):
        param.grad = error_fn(param.detach()).clone()
        e_t = param.grad.clone()
        optimizer.extrapolation()
        param.grad = error_fn(param.detach()).clone()
        e_half = param.grad.clone()
        optimizer.step()
        iterates.append(param.detach().clone())
        errors.append((e_t, e_half))
    return iterates, errors


@pytest.mark.parametrize("xi_update", ["step", "both"])
def test_kp_zero_matches_extra_sgd(xi_update):
    """Kp = 0, ema_nu = 0: ExtraNuPI is ExtraSGD with lr * Ki."""
    lr, Ki = 0.05, 2.0
    p1 = torch.zeros(3, requires_grad=True)
    p2 = torch.zeros(3, requires_grad=True)
    it1, _ = run_extragradient(ExtraNuPI([p1], lr=lr, Kp=0.0, Ki=Ki, xi_update=xi_update, maximize=True), p1, 20)
    it2, _ = run_extragradient(ExtraSGD([p2], lr=lr * Ki, maximize=True), p2, 20)
    for a, b in zip(it1, it2):
        assert torch.allclose(a, b, atol=1e-7)


def _reference(lr, Kp, Ki, nu, xi_update, num_steps):
    """Hand-written ExtraNuPI recurrence (maximize=True)."""
    lam = torch.zeros(3)
    xi = None
    out = []

    def delta(e, xi):
        d = Ki * e
        if xi is not None:
            d = d + Kp * (1 - nu) * (e - xi)
        return lr * d

    def commit(e, xi):
        return e.clone() if xi is None else nu * xi + (1 - nu) * e

    for _ in range(num_steps):
        e_t = error_fn(lam)
        lam_half = lam + delta(e_t, xi)
        if xi_update == "both":
            xi = commit(e_t, xi)
        e_half = error_fn(lam_half)
        lam = lam + delta(e_half, xi)
        xi = commit(e_half, xi)
        out.append(lam.clone())
    return out


@pytest.mark.parametrize("xi_update", ["step", "both"])
@pytest.mark.parametrize(("Kp", "Ki", "nu"), [(1.0, 1.0, 0.0), (5.0, 1.0, 0.0), (1.0, 0.5, 0.9)])
def test_matches_reference_recurrence(xi_update, Kp, Ki, nu):
    lr, num_steps = 0.05, 15
    p = torch.zeros(3, requires_grad=True)
    optimizer = ExtraNuPI([p], lr=lr, Kp=Kp, Ki=Ki, ema_nu=nu, xi_update=xi_update, maximize=True)
    iterates, _ = run_extragradient(optimizer, p, num_steps)
    for a, b in zip(iterates, _reference(lr, Kp, Ki, nu, xi_update, num_steps)):
        assert torch.allclose(a, b, atol=1e-6)


def test_step_and_both_differ_when_kp_positive():
    p1 = torch.zeros(3, requires_grad=True)
    p2 = torch.zeros(3, requires_grad=True)
    it1, _ = run_extragradient(ExtraNuPI([p1], lr=0.05, Kp=2.0, xi_update="step", maximize=True), p1, 5)
    it2, _ = run_extragradient(ExtraNuPI([p2], lr=0.05, Kp=2.0, xi_update="both", maximize=True), p2, 5)
    assert not torch.allclose(it1[-1], it2[-1])


def test_state_dict_roundtrip():
    p = torch.zeros(3, requires_grad=True)
    optimizer = ExtraNuPI([p], lr=0.05, Kp=1.0, maximize=True)
    run_extragradient(optimizer, p, 3)
    new_optimizer = ExtraNuPI([p], lr=0.05, Kp=1.0, maximize=True)
    new_optimizer.load_state_dict(optimizer.state_dict())
    assert torch.allclose(new_optimizer.state[p]["xi"], optimizer.state[p]["xi"])


@pytest.mark.parametrize(
    "kwargs",
    [{"lr": -1.0}, {"Kp": -1.0}, {"Kp": 0.0, "Ki": 0.0}, {"ema_nu": 1.0}, {"xi_update": "never"}],
)
def test_invalid_arguments(kwargs):
    p = torch.zeros(3, requires_grad=True)
    with pytest.raises(ValueError):
        ExtraNuPI([p], **{"lr": 0.1, **kwargs})


class _ToyCMP(cooper.ConstrainedMinimizationProblem):
    """min ||x||^2  s.t.  x_0 + x_1 = 1  (solution x = (0.5, 0.5), lambda = -1)."""

    def __init__(self):
        super().__init__()
        self.constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.EQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
            multiplier=cooper.multipliers.DenseMultiplier(num_constraints=1),
        )

    def compute_cmp_state(self, x):
        violation = (x[0] + x[1] - 1.0).reshape(1)
        return cooper.CMPState(
            loss=(x**2).sum(), observed_constraints={self.constraint: cooper.ConstraintState(violation=violation)}
        )


@pytest.mark.parametrize("xi_update", ["step", "both"])
def test_with_extrapolation_constrained_optimizer(xi_update):
    """ExtraNuPI passes the extrapolation sanity checks and solves a toy problem."""
    x = torch.tensor([2.0, -1.0], requires_grad=True)
    cmp = _ToyCMP()
    optimizer = cooper.optim.ExtrapolationConstrainedOptimizer(
        cmp=cmp,
        primal_optimizers=cooper.optim.ExtraSGD([x], lr=0.05),
        dual_optimizers=ExtraNuPI(cmp.dual_parameters(), lr=0.05, Kp=1.0, xi_update=xi_update, maximize=True),
    )
    for _ in range(2000):
        optimizer.roll(compute_cmp_state_kwargs={"x": x})
    assert torch.allclose(x.detach(), torch.tensor([0.5, 0.5]), atol=1e-3)
