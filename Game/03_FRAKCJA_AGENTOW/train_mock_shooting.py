"""
Train two ANFIS models on a simple shooting mock:
1) Barrel regressor: predict barrel delta to aim at a shootable enemy.
2) Shoot classifier (regression): predict label in {-1, 0, 1}.
"""

from __future__ import annotations

import os
import random
from typing import List

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from ANFISDQN import ANFISDQN
from mock_shooting import (
    _barrel_target,
    _features_for_scene,
    _rand_scene,
    _shoot_label,
    DEFAULT_BARREL_SPIN,
    DEFAULT_HALF_ANGLE,
    DEFAULT_MAX_RANGE,
)


matplotlib.use("Agg")

def _build_dataset(n_samples: int):
    scenes = [_rand_scene() for _ in range(n_samples)]
    feats = [_features_for_scene(s, DEFAULT_MAX_RANGE, DEFAULT_HALF_ANGLE)[0] for s in scenes]
    y_barrel = [_barrel_target(s, DEFAULT_MAX_RANGE, DEFAULT_HALF_ANGLE, DEFAULT_BARREL_SPIN) for s in scenes]
    y_shoot = [_shoot_label(s, DEFAULT_MAX_RANGE, DEFAULT_HALF_ANGLE) for s in scenes]
    return np.stack(feats), np.array(y_barrel, np.float32), np.array(y_shoot, np.float32)


def train_models(
    n_samples: int = 2_000,
    batch_size: int = 128,
    epochs: int = 200,
    seed: int = 1,
):
    #TODO mozna dac jakies dobre labele za to ze odkrywa czolgi przeciwnika, ale tez trzeba by wygenerowac dobre dane
    #TODO mozna dodac skanowanie z historia maja wieze
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x, y_barrel, y_shoot = _build_dataset(n_samples)
    x_t = torch.from_numpy(x)
    yb_t = torch.from_numpy(y_barrel).unsqueeze(1)
    ys_t = torch.from_numpy(y_shoot).unsqueeze(1)

    barrel_model = ANFISDQN(n_inputs=x.shape[1], n_rules=16, n_actions=1).train()
    shoot_model = ANFISDQN(n_inputs=x.shape[1], n_rules=16, n_actions=1).train()

    opt_b = torch.optim.Adam(barrel_model.parameters(), lr=1e-3)
    opt_s = torch.optim.Adam(shoot_model.parameters(), lr=1e-3)

    loss_b_hist: List[float] = []
    loss_s_hist: List[float] = []

    for epoch in range(1, epochs + 1):
        idx = torch.randperm(x_t.size(0))
        epoch_loss_b = 0.0
        epoch_loss_s = 0.0
        batch_count = 0
        for start in range(0, x_t.size(0), batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = x_t[batch_idx]
            yb = yb_t[batch_idx]
            ys = ys_t[batch_idx]

            pred_b = barrel_model(xb)
            loss_b = F.mse_loss(pred_b, yb)
            opt_b.zero_grad(set_to_none=True)
            loss_b.backward()
            opt_b.step()

            pred_s = shoot_model(xb)
            loss_s = F.mse_loss(pred_s, ys)
            opt_s.zero_grad(set_to_none=True)
            loss_s.backward()
            opt_s.step()

            epoch_loss_b += float(loss_b.item())
            epoch_loss_s += float(loss_s.item())
            batch_count += 1

        epoch_loss_b /= max(batch_count, 1)
        epoch_loss_s /= max(batch_count, 1)
        loss_b_hist.append(epoch_loss_b)
        loss_s_hist.append(epoch_loss_s)
        print(f"epoch {epoch}: barrel_loss={epoch_loss_b:.4f} shoot_loss={epoch_loss_s:.4f}")

    _save_plots(
        x_t=x_t,
        y_shoot=y_shoot,
        shoot_model=shoot_model,
        loss_b=loss_b_hist,
        loss_s=loss_s_hist,
    )

    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    barrel_path = os.path.join(out_dir, "anfis_barrel_model.pt")
    shoot_path = os.path.join(out_dir, "anfis_shoot_model.pt")
    torch.save(barrel_model.state_dict(), barrel_path)
    torch.save(shoot_model.state_dict(), shoot_path)
    print(f"saved barrel model: {barrel_path}")
    print(f"saved shoot model: {shoot_path}")

    return barrel_model, shoot_model


def _save_plots(
    x_t: torch.Tensor,
    y_shoot: np.ndarray,
    shoot_model: torch.nn.Module,
    loss_b: List[float],
    loss_s: List[float],
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    out_dir = os.path.join(os.path.dirname(__file__), "training_reports")
    os.makedirs(out_dir, exist_ok=True)

    shoot_model.eval()
    with torch.no_grad():
        pred = shoot_model(x_t).squeeze(1).cpu().numpy()
    pred_labels = np.clip(np.rint(pred), -1.0, 1.0).astype(np.int8)
    true_labels = np.clip(np.rint(y_shoot), -1.0, 1.0).astype(np.int8)

    label_order = [-1, 0, 1]
    cm = confusion_matrix(true_labels, pred_labels, labels=label_order)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=label_order).plot(
        ax=ax, cmap="Blues", colorbar=True, values_format="d"
    )
    ax.set_title("Shoot Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mock_shooting_confusion.png"), dpi=150)
    plt.close(fig)

    loss_df = pd.DataFrame({"barrel_loss": loss_b, "shoot_loss": loss_s})
    ax = loss_df.plot(figsize=(6, 4), title="Mock Shooting Loss", grid=True, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(out_dir, "mock_shooting_loss.png"), dpi=150)
    plt.close(ax.figure)


if __name__ == "__main__":
    train_models()
