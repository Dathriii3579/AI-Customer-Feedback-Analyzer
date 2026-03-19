"""
train.py  –  Multi-Task Training Loop (CSAT + CES)
====================================================
Usage:
    python train.py

Adjust CONFIG dict below to change hyperparameters, dataset path, etc.
All checkpoints are saved to ./checkpoints/.
"""

import os
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from data_prep import build_dataloaders, CSAT_LABEL_MAP, CES_LABEL_MAP
from model import MultiTaskReviewModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    # Dataset
    "csv_path"   : "/Users/vedh/Desktop/csat proj  2/7817_1.csv",
    "max_rows"   : None,          # None = use full dataset
    "max_len"    : 256,
    "batch_size" : 16,
    "num_workers": 0,

    # Model
    "model_name" : "bert-base-uncased",
    "dropout"    : 0.3,

    # Training
    "epochs"         : 5,
    "lr"             : 2e-5,
    "weight_decay"   : 0.01,
    "warmup_ratio"   : 0.1,       # fraction of total steps used for LR warmup
    "alpha_csat"     : 1.0,       # CSAT loss weight
    "alpha_ces"      : 1.0,       # CES  loss weight
    "grad_clip"      : 1.0,
    "seed"           : 42,

    # Output
    "checkpoint_dir" : "./checkpoints",
    "best_ckpt_name" : "best_model.pt",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, csat_criterion, ces_criterion, alpha, beta):
    """
    Runs one pass over `loader` and returns loss + F1 for both tasks.
    """
    model.eval()
    total_loss = 0.0
    all_csat_preds, all_csat_true = [], []
    all_ces_preds,  all_ces_true  = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            csat_lbls  = batch["csat_label"].to(device)
            ces_lbls   = batch["ces_label"].to(device)

            csat_logits, ces_logits = model(input_ids, attn_mask)

            loss = (alpha * csat_criterion(csat_logits, csat_lbls) +
                    beta  * ces_criterion(ces_logits,  ces_lbls))
            total_loss += loss.item()

            all_csat_preds.extend(csat_logits.argmax(dim=-1).cpu().tolist())
            all_csat_true.extend(csat_lbls.cpu().tolist())
            all_ces_preds.extend(ces_logits.argmax(dim=-1).cpu().tolist())
            all_ces_true.extend(ces_lbls.cpu().tolist())

    avg_loss    = total_loss / len(loader)
    csat_acc    = accuracy_score(all_csat_true, all_csat_preds)
    ces_acc     = accuracy_score(all_ces_true,  all_ces_preds)
    csat_f1     = f1_score(all_csat_true, all_csat_preds, average="weighted", zero_division=0)
    ces_f1      = f1_score(all_ces_true,  all_ces_preds,  average="weighted", zero_division=0)

    return {
        "loss"    : avg_loss,
        "csat_acc": csat_acc,
        "csat_f1" : csat_f1,
        "ces_acc" : ces_acc,
        "ces_f1"  : ces_f1,
    }


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    logger.info("Building DataLoaders…")
    train_loader, val_loader, test_loader, tokenizer = build_dataloaders(
        csv_path    = CONFIG["csv_path"],
        model_name  = CONFIG["model_name"],
        max_len     = CONFIG["max_len"],
        batch_size  = CONFIG["batch_size"],
        max_rows    = CONFIG["max_rows"],
        num_workers = CONFIG["num_workers"],
    )

    # ── Class weights (from training split) ──────────────────────────────
    # Pulls CSAT + CES labels directly out of the DataLoader so we don't
    # have to re-read the CSV. Quick scan — no GPU involved here.
    logger.info("Computing class weights from training data…")
    all_csat_train, all_ces_train = [], []
    for batch in train_loader:
        all_csat_train.extend(batch["csat_label"].tolist())
        all_ces_train.extend(batch["ces_label"].tolist())

    import numpy as np
    csat_classes = np.array(sorted(set(all_csat_train)))   # [0, 1, 2]
    ces_classes  = np.array(sorted(set(all_ces_train)))    # [0, 1]

    csat_weights = compute_class_weight(
        class_weight="balanced",
        classes=csat_classes,
        y=all_csat_train
    )
    ces_weights = compute_class_weight(
        class_weight="balanced",
        classes=ces_classes,
        y=all_ces_train
    )

    csat_weight_tensor = torch.tensor(csat_weights, dtype=torch.float).to(device)
    ces_weight_tensor  = torch.tensor(ces_weights,  dtype=torch.float).to(device)
    logger.info(f"CSAT class weights: { {c: f'{w:.3f}' for c, w in zip(csat_classes, csat_weights)} }")
    logger.info(f"CES  class weights: { {c: f'{w:.3f}' for c, w in zip(ces_classes,  ces_weights)} }")

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {CONFIG['model_name']}")
    model = MultiTaskReviewModel(
        model_name = CONFIG["model_name"],
        dropout    = CONFIG["dropout"]
    ).to(device)

    # ── Loss functions (class-weighted) ──────────────────────────────────
    csat_criterion = nn.CrossEntropyLoss(weight=csat_weight_tensor)
    ces_criterion  = nn.CrossEntropyLoss(weight=ces_weight_tensor)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"],
                      weight_decay=CONFIG["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    # ── Main loop ─────────────────────────────────────────────────────────
    alpha = CONFIG["alpha_csat"]
    beta  = CONFIG["alpha_ces"]

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running_loss = 0.0
        all_csat_preds, all_csat_true = [], []
        all_ces_preds,  all_ces_true  = [], []

        for step, batch in enumerate(train_loader, 1):
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            csat_lbls  = batch["csat_label"].to(device)
            ces_lbls   = batch["ces_label"].to(device)

            optimizer.zero_grad()
            csat_logits, ces_logits = model(input_ids, attn_mask)

            loss = (alpha * csat_criterion(csat_logits, csat_lbls) +
                    beta  * ces_criterion(ces_logits,  ces_lbls))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            all_csat_preds.extend(csat_logits.argmax(-1).cpu().tolist())
            all_csat_true.extend(csat_lbls.cpu().tolist())
            all_ces_preds.extend(ces_logits.argmax(-1).cpu().tolist())
            all_ces_true.extend(ces_lbls.cpu().tolist())

            if step % 50 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(train_loader)} "
                    f"| Loss {running_loss/step:.4f}"
                )

        # ── Validation ────────────────────────────────────────────────────
        val_metrics = evaluate(
            model, val_loader, device,
            csat_criterion, ces_criterion, alpha, beta
        )

        train_csat_f1 = f1_score(all_csat_true, all_csat_preds,
                                 average="weighted", zero_division=0)
        train_ces_f1  = f1_score(all_ces_true,  all_ces_preds,
                                 average="weighted", zero_division=0)

        logger.info(
            f"\n── Epoch {epoch}/{CONFIG['epochs']} ──\n"
            f"  Train loss    : {running_loss/len(train_loader):.4f}\n"
            f"  Train CSAT F1 : {train_csat_f1:.4f}\n"
            f"  Train CES  F1 : {train_ces_f1:.4f}\n"
            f"  Val   loss    : {val_metrics['loss']:.4f}\n"
            f"  Val   CSAT F1 : {val_metrics['csat_f1']:.4f} (acc {val_metrics['csat_acc']:.4f})\n"
            f"  Val   CES  F1 : {val_metrics['ces_f1']:.4f} (acc {val_metrics['ces_acc']:.4f})"
        )

        history.append({
            "epoch"       : epoch,
            "train_loss"  : running_loss / len(train_loader),
            "val_loss"    : val_metrics["loss"],
            "val_csat_f1" : val_metrics["csat_f1"],
            "val_ces_f1"  : val_metrics["ces_f1"],
        })

        # ── Save best ──────────────────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config"     : CONFIG,
            }, ckpt_dir / CONFIG["best_ckpt_name"])
            logger.info(f"  ✓ Best checkpoint saved (val_loss={best_val_loss:.4f})")

    # ── Test evaluation ───────────────────────────────────────────────────
    logger.info("\nLoading best checkpoint for final test evaluation…")
    ckpt = torch.load(ckpt_dir / CONFIG["best_ckpt_name"], map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(
        model, test_loader, device,
        csat_criterion, ces_criterion, alpha, beta
    )

    logger.info(
        f"\n══════════════════ TEST RESULTS ══════════════════\n"
        f"  CSAT Accuracy : {test_metrics['csat_acc']:.4f}\n"
        f"  CSAT F1       : {test_metrics['csat_f1']:.4f}\n"
        f"  CES  Accuracy : {test_metrics['ces_acc']:.4f}\n"
        f"  CES  F1       : {test_metrics['ces_f1']:.4f}\n"
        f"═══════════════════════════════════════════════════"
    )

    # ── Save training history ─────────────────────────────────────────────
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump({"config": CONFIG, "history": history, "test": test_metrics}, f, indent=2)

    logger.info("Done. History saved to checkpoints/training_history.json")


if __name__ == "__main__":
    train()
