"""
ANFIS-style network for Q-value approximation.

This module provides a readable and numerically stable fuzzy network:
- learnable membership functions,
- rule firing strengths,
- first-order Sugeno consequents,
- weighted aggregation to Q-values.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMembership(nn.Module):
    """Base class for membership functions used by fuzzy rules."""

    def __init__(
        self,
        n_rules: int,
        n_inputs: int,
        input_min: float,
        input_max: float,
    ):
        super().__init__()
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.input_min = input_min
        self.input_max = input_max

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GaussianMembership(BaseMembership):
    """Gaussian membership: mu(x) = exp(-0.5 * ((x - c) / sigma)^2)."""

    def __init__(
        self,
        n_rules: int,
        n_inputs: int,
        input_min: float,
        input_max: float,
    ):
        super().__init__(n_rules, n_inputs, input_min, input_max)

        centers = torch.linspace(input_min, input_max, n_rules)
        centers = centers.unsqueeze(1).repeat(1, n_inputs)

        init_sigma = max((input_max - input_min) * 0.35, 0.15)

        self.centers = nn.Parameter(centers)
        self.raw_sigma = nn.Parameter(torch.full((n_rules, n_inputs), init_sigma))

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(1)
        sigma = F.softplus(self.raw_sigma) + 1e-3
        z = (x_expanded - self.centers) / sigma
        return -0.5 * z.pow(2)


class BellMembership(BaseMembership):
    """Generalized bell membership."""

    def __init__(
        self,
        n_rules: int,
        n_inputs: int,
        input_min: float,
        input_max: float,
    ):
        super().__init__(n_rules, n_inputs, input_min, input_max)

        centers = torch.linspace(input_min, input_max, n_rules)
        centers = centers.unsqueeze(1).repeat(1, n_inputs)

        init_a = max((input_max - input_min) * 0.25, 0.1)

        self.c = nn.Parameter(centers)
        self.raw_a = nn.Parameter(torch.full((n_rules, n_inputs), init_a))
        self.raw_b = nn.Parameter(torch.full((n_rules, n_inputs), 1.5))

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(1)
        a = F.softplus(self.raw_a) + 1e-3
        b = F.softplus(self.raw_b) + 0.1
        scaled = ((x_expanded - self.c) / a).abs().pow(2.0 * b)
        return -torch.log1p(scaled + 1e-8)


class TriangularMembership(BaseMembership):
    """Triangular membership function with learnable ordered vertices."""

    def __init__(
        self,
        n_rules: int,
        n_inputs: int,
        input_min: float,
        input_max: float,
    ):
        super().__init__(n_rules, n_inputs, input_min, input_max)

        centers = torch.linspace(input_min, input_max, n_rules)
        centers = centers.unsqueeze(1).repeat(1, n_inputs)

        spacing = max((input_max - input_min) / max(n_rules - 1, 1), 0.05)
        self.left = nn.Parameter(centers - spacing)
        self.raw_center_gap = nn.Parameter(torch.full((n_rules, n_inputs), spacing))
        self.raw_right_gap = nn.Parameter(torch.full((n_rules, n_inputs), spacing))

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(1)

        left = self.left
        center = left + F.softplus(self.raw_center_gap) + 1e-4
        right = center + F.softplus(self.raw_right_gap) + 1e-4

        left_slope = (x_expanded - left) / (center - left + 1e-8)
        right_slope = (right - x_expanded) / (right - center + 1e-8)

        mu = torch.clamp(torch.minimum(left_slope, right_slope), 0.0, 1.0)
        return torch.log(mu + 1e-8)


MEMBERSHIP_MAP = {
    "gaussian": GaussianMembership,
    "bell": BellMembership,
    "triangular": TriangularMembership,
}


class ANFISDQN(nn.Module):
    """
    ANFIS-style Q-network.

    Args:
        n_inputs: Number of input features.
        n_rules: Number of fuzzy rules.
        n_actions: Number of discrete actions.
        mf_type: Type of membership function.
        input_min: Lower bound of normalized inputs.
        input_max: Upper bound of normalized inputs.
    """

    def __init__(
        self,
        n_inputs: int,
        n_rules: int,
        n_actions: int,
        mf_type: Literal["gaussian", "bell", "triangular"] = "gaussian",
        input_min: float = 0.0,
        input_max: float = 1.0,
    ):
        super().__init__()

        if n_inputs <= 0:
            raise ValueError("n_inputs must be > 0")
        if n_rules <= 0:
            raise ValueError("n_rules must be > 0")
        if n_actions <= 0:
            raise ValueError("n_actions must be > 0")
        if mf_type not in MEMBERSHIP_MAP:
            raise ValueError(f"Unsupported mf_type: {mf_type}")

        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.n_actions = n_actions

        membership_cls = MEMBERSHIP_MAP[mf_type]
        self.membership = membership_cls(n_rules, n_inputs, input_min, input_max)

        # Consequent part (first-order Sugeno):
        # rule_output = w_rule * x + b_rule
        self.consequent_weights = nn.Parameter(
            torch.empty(n_rules, n_actions, n_inputs)
        )
        self.consequent_bias = nn.Parameter(torch.zeros(n_rules, n_actions))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.consequent_weights)
        nn.init.zeros_(self.consequent_bias)

    def _rule_strengths(self, x: torch.Tensor) -> torch.Tensor:
        # log_mu: [batch, rules, inputs]
        log_mu = self.membership.log_membership(x)

        # Mean instead of product to avoid numerical underflow with many inputs.
        rule_logits = log_mu.mean(dim=2)
        return torch.softmax(rule_logits, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape [batch, n_actions]."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.float()

        strengths = self._rule_strengths(x)  # [B, R]
        rule_outputs = (
            torch.einsum("bi,rai->bra", x, self.consequent_weights)
            + self.consequent_bias
        )  # [B, R, A]

        q_values = torch.einsum("br,bra->ba", strengths, rule_outputs)
        return q_values

    def firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized rule activations [batch, n_rules]."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self._rule_strengths(x.float())

