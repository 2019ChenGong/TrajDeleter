from typing import List, Optional, Union, cast

import torch
from torch import nn

from .base import ContinuousQFunction, DiscreteQFunction


def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    # reduction beased on expectation
    mean = y.mean(dim=-1)
    if reduction == "min":
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


class EnsembleQFunction(nn.Module):  # type: ignore
    _action_size: int
    _q_funcs: nn.ModuleList
    _bootstrap: bool

    def __init__(
        self,
        q_funcs: Union[List[DiscreteQFunction], List[ContinuousQFunction]],
        bootstrap: bool = False,
    ):
        super().__init__()
        self._action_size = q_funcs[0].action_size
        self._q_funcs = nn.ModuleList(q_funcs)
        self._bootstrap = bootstrap and len(q_funcs) > 1

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        use_independent_target: bool = False,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_independent_target:
            assert q_tp1.ndim == 3
        else:
            assert q_tp1.ndim == 2

        if self._bootstrap and masks is not None:
            assert masks.shape == (len(self._q_funcs), obs_t.shape[0], 1,), (
                "Invalid mask shape is detected. "
                f"mask_size must be {len(self._q_funcs)}."
            )

        td_sum = torch.tensor(0.0, dtype=torch.float32, device=obs_t.device)
        for i, q_func in enumerate(self._q_funcs):
            if use_independent_target:
                target = q_tp1[i]
            else:
                target = q_tp1

            loss = q_func.compute_error(
                obs_t, act_t, rew_tp1, target, ter_tp1, gamma, reduction="none"
            )

            if self._bootstrap:
                if masks is None:
                    mask = torch.randint(0, 2, loss.shape, device=obs_t.device)
                else:
                    mask = masks[i]
                loss *= mask.float()
                td_sum += loss.sum() / (mask.sum().float() + 1e-10)
            else:
                td_sum += loss.mean()

        return td_sum

    def _compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        values_list: List[torch.Tensor] = []
        for q_func in self._q_funcs:
            target = q_func.compute_target(x, action)
            values_list.append(target.reshape(1, x.shape[0], -1))

        values = torch.cat(values_list, dim=0)

        if action is None:
            # mean Q function
            if values.shape[2] == self._action_size:
                return _reduce_ensemble(values, reduction)
            # distributional Q function
            n_q_funcs = values.shape[0]
            values = values.view(n_q_funcs, x.shape[0], self._action_size, -1)
            return _reduce_quantile_ensemble(values, reduction)

        if values.shape[2] == 1:
            return _reduce_ensemble(values, reduction, lam=lam)

        return _reduce_quantile_ensemble(values, reduction, lam=lam)

    @property
    def q_funcs(self) -> nn.ModuleList:
        return self._q_funcs

    @property
    def bootstrap(self) -> bool:
        return self._bootstrap


class EnsembleDiscreteQFunction(EnsembleQFunction):
    def forward(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x).view(1, x.shape[0], self._action_size))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, reduction, lam)


class EnsembleContinuousQFunction(EnsembleQFunction):
    def forward(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, reduction, lam)
