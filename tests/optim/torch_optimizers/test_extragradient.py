# Copyright (C) 2025 The Cooper Developers.
# Licensed under the MIT License.

import torch

from cooper.optim import ExtraAdam, ExtraSGD


def _quadratic_steps(optimizer, param, num_steps=3):
    for _ in range(num_steps):
        optimizer.zero_grad()
        (param**2).sum().backward()
        optimizer.extrapolation()
        optimizer.zero_grad()
        (param**2).sum().backward()
        optimizer.step()


def test_extra_sgd_load_state_dict_roundtrip():
    """Regression test: `ExtraSGD.__setstate__` used to call
    `super(torch.optim.SGD, self)`, raising a TypeError in `load_state_dict`."""
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = ExtraSGD([param], lr=0.1, momentum=0.9)
    _quadratic_steps(optimizer, param)

    state_dict = optimizer.state_dict()
    new_optimizer = ExtraSGD([param], lr=0.1, momentum=0.9)
    new_optimizer.load_state_dict(state_dict)  # must not raise

    buffer = optimizer.state[param]["momentum_buffer"]
    assert torch.allclose(new_optimizer.state[param]["momentum_buffer"], buffer)
    assert new_optimizer.param_groups[0]["nesterov"] is False


def test_extra_adam_load_state_dict_roundtrip():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = ExtraAdam([param], lr=0.1)
    _quadratic_steps(optimizer, param)
    new_optimizer = ExtraAdam([param], lr=0.1)
    new_optimizer.load_state_dict(optimizer.state_dict())
    assert torch.allclose(new_optimizer.state[param]["exp_avg"], optimizer.state[param]["exp_avg"])
