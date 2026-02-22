"""
    Train two ANFIS models on a simple shooting mock:
    1) Barrel regressor: predict barrel delta to aim at a shootable enemy.
    2) Shoot classifier (logits): predict label in {0, 1}.
"""

from __future__ import annotations

import os
import random
from typing import List

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ANFISDQN import ANFISDQN
from mock_shooting import (
    _barrel_target,
    _features_for_scene,
    _rand_scene,
    _shoot_label,
    DEFAULT_BARREL_STEP,
    DEFAULT_HALF_ANGLE,
    DEFAULT_MAX_RANGE,
)

matplotlib.use("Agg")

def _build_dataset(n_samples: int):
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    scenes = [_rand_scene() for _ in range(n_samples)]
    feats = [_features_for_scene(s)[0] for s in scenes]
    y_barrel = [_barrel_target(s, DEFAULT_BARREL_STEP) for s in scenes]
    y_shoot = [_shoot_label(s, DEFAULT_MAX_RANGE, DEFAULT_HALF_ANGLE) for s in scenes]
    return np.stack(feats), np.array(y_barrel, np.float32), np.array(y_shoot, np.float32)


def train_models(
    n_samples: int = 20_000,
    batch_size: int = 128,
    epochs: int = 200,
    seed: int = 1,
    val_ratio: float = 0.2,
):
    #TODO mozna dac jakies dobre labele za to ze odkrywa czolgi przeciwnika, ale tez trzeba by wygenerowac dobre dane
    #TODO mozna dodac skanowanie z historia maja wieze - to juz calkiem ambitne. Na razie bez historii.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    x, y_barrel, y_shoot = _build_dataset(n_samples)
    n_val = int(n_samples * val_ratio)
    if n_val < 1:
        raise ValueError(
            f"Validation split too small for n_samples={n_samples}, val_ratio={val_ratio}. "
            f"Increase n_samples or val_ratio."
        )
    if n_val >= n_samples:
        raise ValueError(
            f"Validation split too large for n_samples={n_samples}: "
            f"computed n_val={n_val}. Increase n_samples or lower val_ratio."
        )
    perm = np.random.permutation(n_samples)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"dataset split: total={n_samples} train={len(train_idx)} val={len(val_idx)}")

    x_train = x[train_idx]
    yb_train = y_barrel[train_idx]
    ys_train = y_shoot[train_idx]
    x_val = x[val_idx]
    ys_val = y_shoot[val_idx]

    x_t = torch.from_numpy(x_train)
    yb_t = torch.from_numpy(yb_train).unsqueeze(1)
    ys_t = torch.from_numpy(ys_train).unsqueeze(1)
    positive_count = float(ys_train.sum())
    negative_count = float(len(ys_train) - positive_count)
    if positive_count <= 0.0:
        raise ValueError("Shoot dataset has no positive labels; cannot train BCE classifier.")
    if negative_count <= 0.0:
        raise ValueError("Shoot dataset has no negative labels; cannot train BCE classifier.")
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    loss_fn_s = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            if not torch.isfinite(pred_s).all():
                raise ValueError("Non-finite shoot logits during training")
            loss_s = loss_fn_s(pred_s, ys)
            if not torch.isfinite(loss_s):
                raise ValueError("Non-finite shoot loss during training")
            opt_s.zero_grad(set_to_none=True)
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(shoot_model.parameters(), max_norm=1.0)
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
        x_val=x_val,
        y_shoot_val=ys_val,
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
    x_val: np.ndarray,
    y_shoot_val: np.ndarray,
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
    x_val_t = torch.from_numpy(x_val)
    with torch.no_grad():
        pred_values = shoot_model(x_val_t).squeeze(1)
    pred_labels = (pred_values >= 0.0).numpy().astype(np.int8)
    true_labels = np.clip(np.rint(y_shoot_val), 0.0, 1.0).astype(np.int8)

    label_order = [0, 1]
    cm = confusion_matrix(true_labels, pred_labels, labels=label_order)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=label_order).plot(
        ax=ax, cmap="Blues", colorbar=True, values_format="d"
    )
    ax.set_title("Shoot Confusion Matrix (Validation)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mock_shooting_confusion.png"), dpi=150)
    plt.close(fig)

    loss_df = pd.DataFrame({"barrel_loss": loss_b, "shoot_loss": loss_s})
    ax = loss_df.plot(figsize=(6, 4), title="Mock Shooting Loss", grid=True, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(out_dir, "mock_shooting_loss.png"), dpi=150)
    plt.close(ax.figure)


if __name__ == "__main__":
    train_models()
