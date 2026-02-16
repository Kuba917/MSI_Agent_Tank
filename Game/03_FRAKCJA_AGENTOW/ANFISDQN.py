"""
Layers:
    1. Fuzzification   — Gaussian MFs with linspace-initialized centers
    2. Rule firing      — product T-norm
    3. Normalization    — normalized firing strengths
    4. Consequent       — first-order Sugeno (linear per rule)
    5. Aggregation      — weighted sum → (batch, n_outputs)
"""

import torch
import torch.nn as nn
from typing import Literal


class GaussianMF(nn.Module):
    """μ(x) = exp(-(x - c)² / (2σ²)), centers initialized via linspace."""

    def __init__(self, n_rules: int, n_inputs: int):
        super().__init__()
        # Uniform spread across [0, 1] — normalize your inputs!
        centers = torch.linspace(0, 1, n_rules).unsqueeze(1).expand(n_rules, n_inputs).clone()
        self.centers = nn.Parameter(centers)
        self.sigmas = nn.Parameter(torch.full((n_rules, n_inputs), 1.0 / n_rules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_inputs) → (batch, n_rules, n_inputs)
        x = x.unsqueeze(1)
        sigma = self.sigmas.abs().clamp(min=1e-4)
        return torch.exp(-((x - self.centers) ** 2) / (2 * sigma ** 2))


class BellMF(nn.Module):
    """Generalized bell: 1 / (1 + |(x-c)/a|^(2b))"""

    def __init__(self, n_rules: int, n_inputs: int):
        super().__init__()
        centers = torch.linspace(0, 1, n_rules).unsqueeze(1).expand(n_rules, n_inputs).clone()
        self.c = nn.Parameter(centers)
        self.a = nn.Parameter(torch.full((n_rules, n_inputs), 1.0 / n_rules))
        self.b = nn.Parameter(torch.full((n_rules, n_inputs), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        a = self.a.abs().clamp(min=1e-4)
        b = self.b.abs().clamp(min=0.5)
        return 1.0 / (1.0 + ((x - self.c) / a).abs().pow(2 * b) + 1e-8)


class TriangularMF(nn.Module):
    """Triangular MF defined by left/center/right."""

    def __init__(self, n_rules: int, n_inputs: int):
        super().__init__()
        spacing = 1.0 / max(n_rules - 1, 1)
        centers = torch.linspace(0, 1, n_rules).unsqueeze(1).expand(n_rules, n_inputs).clone()
        self.left = nn.Parameter(centers - spacing)
        self.center = nn.Parameter(centers)
        self.right = nn.Parameter(centers + spacing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        l, c, r = self.left, self.center, self.right
        # Ensure l < c < r via softplus offsets
        c_safe = l + (c - l).abs() + 1e-6
        r_safe = c_safe + (r - c_safe).abs() + 1e-6
        left_slope = (x - l) / (c_safe - l + 1e-8)
        right_slope = (r_safe - x) / (r_safe - c_safe + 1e-8)
        return torch.clamp(torch.min(left_slope, right_slope), 0.0, 1.0)


MF_TYPES = {
    "gaussian": GaussianMF,
    "bell": BellMF,
    "triangular": TriangularMF,
}

"""
Dobra no i co inna logika dla ciezkiego inna dla lekkiego - moze pierw niech kazdy ma taka sama to by wysoka gamma musiala byc zeby bylo inaczej
Inde Input Semantic	Normalized Value Range	Calculation / Logic
    0	My HP	0.0 to 1.0	Raw HP / 100.0
    1	Has Heavy Ammo	0.0 or 1.0	1.0 if count > 0, else 0.0
    2	Enemy Distance	0.0 to 1.0	min(Distance / 500.0, 1.0)
    3	Enemy Angle	-1.0 to 1.0	Relative Degrees / 180.0
    4	Enemy HP	0.0 to 1.0	Enemy HP / 100.0
    5	Powerup Distance	0.0 to 1.0	min(Distance / 500.0, 1.0)
    6	Powerup Angle	-1.0 to 1.0	Relative Degrees / 180.0
    7	Aimed at Friend	0.0 or 1.0	1.0 if aiming at teammate, else 0.0
    8	Obstacle Distance	0.0 to 1.0	min(Distance / 500.0, 1.0)
    9	Obstacle Angle	-1.0 to 1.0	Relative Degrees / 180.0
    10	Danger Distance	0.0 to 1.0	min(Distance / 500.0, 1.0)
    11	Danger Angle	-1.0 to 1.0	Relative Degrees / 180.0
"""

"""
Poki co ANFIS bedzie uzywac ANDa:
    1. Ale w przyszlosci mozna byc skierowac uwage na ORa -> Wg. Ai moze zostac AND jest bardziej flexible podobno
    2. Czy dac anfisa oddzielnie na kazda akcje
"""

class ANFISDQN(nn.Module):
    """
    Args:
        n_inputs:  Number of input features.
        n_rules:   Number of fuzzy rules.
        n_outputs: Number of outputs (53 for your classification).
        mf_type:   "gaussian", "bell", or "triangular".
    """

    def __init__(self,
            n_inputs: int,
            n_rules: int, 
            n_outputs: int = 1,
            mf_type: Literal["gaussian", "triangular", "bell"] = "gaussian", 
        ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.n_outputs = n_outputs
        self.mf = MF_TYPES[mf_type](n_rules, n_inputs)

        # Layer 4: Consequent
        # (n_rules, n_outputs, n_inputs + 1) — linear per rule per output
        self.consequent = nn.Parameter(torch.randn(n_rules, n_outputs, n_inputs + 1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_inputs) → (batch, n_outputs)"""
        batch = x.size(0)

        # Layer 1: Fuzzify → (batch, n_rules, n_inputs).
        mu = self.mf(x)

        # Layer 2: Rule strengths → (batch, n_rules) -> Poki co jest algebraiczny AND, raczej ORa nie bedziemy za bardzo potrzebowac podobno
        w = mu.prod(dim=2)

        # Layer 3: Normalize -> BarchNorm
        w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-8)

        # Layer 4 + 5: Consequent & aggregate
        x_aug = torch.cat([x, x.new_ones(batch, 1)], dim=1)          # (batch, n_inputs+1)
        f = torch.einsum("bi, roi -> bro", x_aug, self.consequent)    # (batch, n_rules, n_outputs) -> Tu sobie wymnaza tak naprawde tu dochodzi do ANDa
        unnormalized_output = torch.einsum("br, bro -> bo", w_norm, f)               # (batch, n_outputs) -> tu srednia
        return unnormalized_output # tutaj funkcje przynaleznosci I guess

    def firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized rule activations — useful for debugging."""
        mu = self.mf(x)
        w = mu.prod(dim=2)
        return w / (w.sum(dim=1, keepdim=True) + 1e-8)


"""
    While you can send any float value, the engine clamps these values based on the specific tank type you are controlling; - te 5 akcji mozemy wykonac

    1. barrel_rotation_angle (float) purpose: Rotates the turret relative to the hull.
        Engine Limits (per tick):
        LIGHT: +/- 90.0 degrees
        HEAVY: +/- 70.0 degrees
        SNIPER: +/- 100.0 degrees
    2. heading_rotation_angle (float)
        Purpose: Rotates the entire tank hull.
        Engine Limits (per tick):
        LIGHT: +/- 70.0 degrees
        HEAVY: +/- 30.0 degrees
        SNIPER: +/- 45.0 degrees
    3. move_speed (float)
        Purpose: Forward (+) or Backward (-) movement.
        Engine Limits (units per tick):
        LIGHT: +/- 5.0
        HEAVY: +/- 1.0
        SNIPER: +/- 3.0
        Note: Agents like your DQN.py often use a +/- 100.0 scale for convenience, but the physics engine will cap the actual movement at the values above.
    4. ammo_to_load (str)
        Possible Values: "HEAVY", "LIGHT", "LONG_DISTANCE", or None.
    5. should_fire (bool)
        Possible Values: True or False.
"""