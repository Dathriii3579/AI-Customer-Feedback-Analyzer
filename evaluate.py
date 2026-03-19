"""
evaluate.py  –  Visualise training results and model evaluation
================================================================
Run after training:
    python evaluate.py

Produces plots saved to ./checkpoints/:
  - training_curves.png  (loss, CSAT F1, CES F1 over epochs)
  - confusion_csat.png
  - confusion_ces.png
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)

from data_prep import build_dataloaders, CSAT_LABEL_MAP, CES_LABEL_MAP
from model import MultiTaskReviewModel, CSAT_CLASSES, CES_CLASSES

CKPT_DIR   = Path("./checkpoints")
BEST_CKPT  = CKPT_DIR / "best_model.pt"
CSV_PATH   = "/Users/vedh/Desktop/csat proj  2/7817_1.csv"


# ─── 1. Training curves ───────────────────────────────────────────────────────

def plot_training_curves():
    history_path = CKPT_DIR / "training_history.json"
    if not history_path.exists():
        print(f"Not found: {history_path}. Train first.")
        return

    with open(history_path) as f:
        data = json.load(f)

    hist  = data["history"]
    ep    = [h["epoch"]       for h in hist]
    tl    = [h["train_loss"]  for h in hist]
    vl    = [h["val_loss"]    for h in hist]
    csat  = [h["val_csat_f1"] for h in hist]
    ces   = [h["val_ces_f1"]  for h in hist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Multi-Task Training Curves", fontsize=14, fontweight="bold")

    axes[0].plot(ep, tl, "o-", label="Train Loss")
    axes[0].plot(ep, vl, "s-", label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(ep, csat, "o-", label="CSAT F1")
    axes[1].plot(ep, ces,  "s-", label="CES  F1")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Weighted F1")
    axes[1].set_title("Validation F1"); axes[1].legend()

    plt.tight_layout()
    out = CKPT_DIR / "training_curves.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close()


# ─── 2. Confusion matrices ────────────────────────────────────────────────────

def plot_confusion_matrices(max_rows=5000):
    if not BEST_CKPT.exists():
        print(f"Checkpoint not found: {BEST_CKPT}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(BEST_CKPT, map_location=device)
    cfg    = ckpt.get("config", {})
    model_name = cfg.get("model_name", "bert-base-uncased")

    model = MultiTaskReviewModel(model_name=model_name)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    _, _, test_loader, _ = build_dataloaders(
        csv_path   = CSV_PATH,
        model_name = model_name,
        max_rows   = max_rows
    )

    all_csat_preds, all_csat_true = [], []
    all_ces_preds,  all_ces_true  = [], []

    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            csat_logits, ces_logits = model(ids, mask)
            all_csat_preds.extend(csat_logits.argmax(-1).cpu().tolist())
            all_csat_true.extend(batch["csat_label"].tolist())
            all_ces_preds.extend(ces_logits.argmax(-1).cpu().tolist())
            all_ces_true.extend(batch["ces_label"].tolist())

    def save_cm(true, pred, labels, title, fname):
        cm   = confusion_matrix(true, pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        out = CKPT_DIR / fname
        plt.savefig(out, dpi=150)
        print(f"Saved → {out}")
        plt.close()

    save_cm(all_csat_true, all_csat_preds,
            CSAT_CLASSES, "CSAT Confusion Matrix", "confusion_csat.png")
    save_cm(all_ces_true,  all_ces_preds,
            CES_CLASSES,  "CES  Confusion Matrix", "confusion_ces.png")

    print("\nCSAT Classification Report:\n",
          classification_report(all_csat_true, all_csat_preds,
                                target_names=CSAT_CLASSES, zero_division=0))
    print("\nCES  Classification Report:\n",
          classification_report(all_ces_true,  all_ces_preds,
                                target_names=CES_CLASSES,  zero_division=0))


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Training Curves ──")
    plot_training_curves()
    print("\n── Confusion Matrices ──")
    plot_confusion_matrices()
