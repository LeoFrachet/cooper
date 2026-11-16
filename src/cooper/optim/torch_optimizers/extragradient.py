# Copyright (C) 2025 The Cooper Developers.
# Licensed under the MIT License.

"""Optimizers based on extra-gradient."""

import math
from collections.abc import Callable, Iterable
from typing import NoReturn, Optional

import torch

# -----------------------------------------------------------------------------
# Implementation of ExtraOptimizers contains minor edits on source code from:
# https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py
# * We add a `maximize` flag to the `ExtraSGD` and `ExtraAdam` classes to allow for
# maximization steps.
# * We slightly modify the docstrings to comply with our style guide.
#
# TODO(juan43ramirez): The implementations below "manually" apply SGD and Adam updates.
# Alternatively, we could carry out updates using functional implementations of SGD and
# Adam from `torch.optim`. This way, we can easily stay up-to-date with the community
# approved implementations of these optimizers.


# -----------------------------------------------------------------------------

#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.


class ExtragradientOptimizer(torch.optim.Optimizer):
    r"""Base class for :class:`torch.optim.Optimizer`\s with an ``extrapolation`` step.

    Args:
        params: an iterable of :class:`torch.Tensor`\s or
            :class:`dict`\s. Specifies what Tensors should be optimized.
        defaults: a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params: Iterable, defaults: dict) -> None:
        super().__init__(params, defaults)
        self.params_copy: list[torch.nn.Parameter] = []

    def update(self, p: torch.Tensor, group: dict) -> NoReturn:
        raise NotImplementedError

    def extrapolation(self) -> None:
        """Performs the extrapolation step and saves a copy of the current
        parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for p in group["params"]:
                u = self.update(p, group)
                if is_empty:
                    # Save the current parameters for the update step. Several
                    # extrapolation step can be made before each update but only
                    # the parameters before the first extrapolation step are
                    # saved.
                    self.params_copy.append(p.data.clone())
                if u is None:
                    continue
                # Update the current parameters
                p.data.add_(u)

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if len(self.params_copy) == 0:
            raise RuntimeError("Need to call extrapolation before calling step.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        i = -1
        for group in self.param_groups:
            for p in group["params"]:
                i += 1
                u = self.update(p, group)
                if u is None:
                    continue
                # Update the parameters saved during the extrapolation step
                p.data = self.params_copy[i].add_(u)

        # Free the old parameters
        self.params_copy = []
        return loss


class ExtraSGD(ExtragradientOptimizer):
    r"""Extrapolation-compatible implementation of SGD with momentum.

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        :cite:p:`sutskever2013initialization` and implementations in some other
        frameworks.

        Considering the specific case of Momentum, the update can be written as:

        .. math::
            \vv_{t+1} = \rho \cdot \vv_t + \nabla_{\vtheta} L(\vtheta_t) \\
            \vtheta_{t+1} = \vtheta_t - \eta \cdot \vv_{t+1},

        where :math:`\vtheta`, :math:`\vv`, :math:`\nabla_{\vtheta} L` and :math:`\rho`
        denote the parameters, velocity, gradient and momentum respectively.

        This is in contrast to :cite:p:`sutskever2013initialization` and
        other frameworks which employ an update of the form:

        .. math::
            \vv_{t+1} &= \rho \cdot \vv_t + \eta \cdot \nabla_{\vtheta} L(\vtheta_t) \\
            \vtheta_{t+1} &= \vtheta_t - \vv_{t+1}.

        The Nesterov version is modified analogously.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter
            groups.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay (L2 penalty).
        dampening: Dampening for momentum.
        nesterov: If ``True``, enables Nesterov momentum.

    Raises:
        ValueError: If the learning rate, momentum, or weight decay are negative.
        ValueError: If Nesterov momentum is enabled while momentum is set to zero or
            dampening is not zero.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
        }
        if nesterov and (momentum == 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        # ExtraSGD does not subclass torch.optim.SGD, so `super(torch.optim.SGD, self)`
        # raised a TypeError (e.g. in `load_state_dict`).
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def update(self, p: torch.Tensor, group: dict) -> Optional[torch.Tensor]:
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]
        maximize = group["maximize"]

        if p.grad is None:
            return None
        d_p = p.grad.data
        if maximize:
            d_p = -d_p
        if weight_decay != 0:
            d_p.add_(weight_decay, p.data)
        if momentum != 0:
            param_state = self.state[p]
            if "momentum_buffer" not in param_state:
                buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                buf.mul_(momentum).add_(d_p)
            else:
                buf = param_state["momentum_buffer"]
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            d_p = d_p.add(momentum, buf) if nesterov else buf

        return -group["lr"] * d_p


class ExtraAdam(ExtragradientOptimizer):
    """Implements the Adam algorithm with an extrapolation step.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr : Learning rate.
        betas: Coefficients used for computing running averages of gradient and
            its square.
        eps : Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay (L2 penalty).
        amsgrad: Flag to use the AMSGrad variant of this algorithm from
            :cite:p:`reddi2018amsgrad`.

    Raises:
        ValueError: If the learning rate or epsilon value is negative.
        ValueError: If the beta parameters are not in the range [0, 1).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def update(self, p: torch.Tensor, group: dict) -> Optional[torch.Tensor]:
        if p.grad is None:
            return None
        grad = p.grad.data
        if group["maximize"]:
            grad = -grad
        if grad.is_sparse:
            raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
        amsgrad = group["amsgrad"]

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state["step"] = 0
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
        beta1, beta2 = group["betas"]

        state["step"] += 1

        if group["weight_decay"] != 0:
            grad = grad.add(group["weight_decay"], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group["eps"])
        else:
            denom = exp_avg_sq.sqrt().add_(group["eps"])

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

        return -step_size * exp_avg / denom


class ExtraNuPI(ExtragradientOptimizer):
    r"""The :py:class:`~cooper.optim.nuPI` controller with an extrapolation step.

    This makes nuPI compatible with
    :py:class:`~cooper.optim.ExtrapolationConstrainedOptimizer`, so that the
    Lagrange multipliers can be updated by a PI controller while the whole
    primal-dual game is solved with extragradient.

    The update is the (SGD-initialized) nuPI update. With error
    :math:`e_t = \nabla_{\vlambda} \Lag` (the constraint violation, for a dual
    optimizer with ``maximize=True``):

    .. math::
        \Delta_t = K_I e_t + K_P (1 - \nu) (e_t - \xi_{t-1}), \qquad
        \xi_t = \nu \xi_{t-1} + (1 - \nu) e_t, \qquad \xi_0 = e_0,

    and the proportional term is omitted at the very first update, so that the
    first step matches SGD with learning rate :math:`\eta K_I`.

    Extragradient calls :py:meth:`update` twice per iteration: once at the
    current point (extrapolation) and once at the extrapolated point (the
    actual step, applied from the current point). ``xi_update`` decides when
    the error memory :math:`\xi` is updated:

    * ``"step"`` (default): only during the actual step. :math:`\xi` then
      tracks the errors measured at the extrapolated points, i.e. the errors
      that drive the actual updates, and the extrapolation uses :math:`\xi`
      without modifying it. The actual steps form a nuPI controller acting on
      the sequence of look-ahead errors.
    * ``"both"``: during both calls, mirroring how
      :py:class:`ExtraSGD` / :py:class:`ExtraAdam` update their momentum
      buffers / moment estimates. The proportional term of the actual step
      then compares the look-ahead error with the error at the current point.

    With ``Kp=0`` and ``ema_nu=0`` both variants reduce exactly to
    :py:class:`ExtraSGD` (no momentum) with learning rate ``lr * Ki``.

    .. note::
        nuPI with :math:`K_P = 1` (and :math:`\nu = 0`) corresponds to optimistic
        gradient ascent, itself an approximation of extragradient
        ("extrapolation from the past"). Combining extragradient with a large
        :math:`K_P` may therefore apply this damping twice.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        weight_decay: Weight decay (applied after the error terms, as in nuPI).
        Kp: Proportional gain (a float).
        Ki: Integral gain (a float).
        ema_nu: EMA coefficient :math:`\nu \in [0, 1)` of the error memory.
        xi_update: ``"step"`` or ``"both"``, see above.
        maximize: Maximize the objective (use ``True`` for dual optimizers).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float,
        weight_decay: float = 0.0,
        Kp: float = 0.0,
        Ki: float = 1.0,
        ema_nu: float = 0.0,
        xi_update: str = "step",
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if Kp < 0.0 or Ki < 0.0:
            raise ValueError(f"Kp and Ki must be non-negative, got Kp={Kp}, Ki={Ki}")
        if Kp == 0.0 and Ki == 0.0:
            raise ValueError("Kp and Ki cannot both be zero.")
        if not 0.0 <= ema_nu < 1.0:
            raise ValueError(f"Invalid ema_nu value: {ema_nu}")
        if xi_update not in {"step", "both"}:
            raise ValueError(f"xi_update must be 'step' or 'both', got {xi_update!r}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "Kp": Kp,
            "Ki": Ki,
            "ema_nu": ema_nu,
            "xi_update": xi_update,
            "maximize": maximize,
        }
        super().__init__(params, defaults)
        self._is_extrapolating = False

    def extrapolation(self) -> None:
        self._is_extrapolating = True
        try:
            super().extrapolation()
        finally:
            self._is_extrapolating = False

    def update(self, p: torch.Tensor, group: dict) -> Optional[torch.Tensor]:
        if p.grad is None:
            return None
        if p.grad.is_sparse:
            raise NotImplementedError("ExtraNuPI does not support sparse gradients.")

        error = p.grad.detach()
        Kp, Ki, ema_nu = group["Kp"], group["Ki"], group["ema_nu"]
        state = self.state[p]
        uses_kp_term = Kp * (1 - ema_nu) != 0

        nupi_update = error.mul(Ki)
        if uses_kp_term and "xi" in state:
            nupi_update.add_((error - state["xi"]).mul(Kp * (1 - ema_nu)))
        if group["weight_decay"] != 0:
            nupi_update.add_(p.data, alpha=-group["weight_decay"] if group["maximize"] else group["weight_decay"])

        commit = (not self._is_extrapolating) or group["xi_update"] == "both"
        if uses_kp_term and commit:
            if "xi" not in state:
                state["xi"] = error.clone()  # xi_0 = e_0 (SGD initialization)
            else:
                state["xi"].mul_(ema_nu).add_(error, alpha=1 - ema_nu)

        alpha = group["lr"] if group["maximize"] else -group["lr"]
        return nupi_update.mul_(alpha)
