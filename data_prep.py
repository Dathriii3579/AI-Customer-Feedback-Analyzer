"""
data_prep.py  –  Amazon + TrustPilot Consumer Reviews
======================================================
Primary dataset (Amazon 7817_1.csv):
    reviews.text        → review body text
    reviews.title       → review title  (prepended to body)
    reviews.rating      → 1-5 star rating  → CSAT label
    reviews.doRecommend → bool             → refines borderline CSAT

Supplementary dataset (TrustPilot trust_pilot_reviews_data_2022_06.csv):
    review_text / review_title  → combined text input
    rating                      → 1-5 star rating → CSAT label

CSAT Labels:
    0 = Dissatisfied  (rating 1–2)
    1 = Neutral       (rating 3, or ambiguous)
    2 = Satisfied     (rating 4–5)

CES Labels (derived from effort/friction keywords in review text):
    0 = Easy
    1 = Difficult
"""

import re
import csv
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Dataset constants ────────────────────────────────────────────────────────

# Column names as they appear in 7817_1.csv
CSV_TEXT_COL      = "reviews.text"
CSV_TITLE_COL     = "reviews.title"
CSV_RATING_COL    = "reviews.rating"
CSV_RECOMMEND_COL = "reviews.doRecommend"

CSAT_LABEL_MAP = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}
CES_LABEL_MAP  = {0: "Easy", 1: "Difficult"}

# ─── CES keyword patterns ────────────────────────────────────────────────────
# Captures effort and friction signals — covers most e-commerce complaint types.

EFFORT_KEYWORDS = [
    r"\b(late|delayed|delay)\b",
    r"\brefund\b",
    r"\breturn\b",
    r"\bcustomer service\b",
    r"\bsupport\b",
    r"\bcomplaint\b",
    r"\bdamaged\b",
    r"\bbroken\b",
    r"\bmissing\b",
    r"\bdefective\b",
    r"\bnever arrived\b",
    r"\bnot delivered\b",
    r"\bpackaging\b",
    r"\bfrustrat\w*\b",
    r"\bdifficult\b",
    r"\bhard to\b",
    r"\bconfus\w*\b",
    r"\bpoor quality\b",
    r"\bdisappoint\w*\b",
    r"\bwrong item\b",
    r"\bcharged\b",
    r"\bovercharged\b",
    r"\bwaited (long|too)\b",
    r"\bwaiting (forever|a long)\b",
    r"\bnot working\b",
    r"\bdoesn'?t work\b",
    r"\bstop(ped)? working\b",
    r"\breplac\w*\b",
    r"\bescalat\w*\b",
    r"\bterrible\b",
    r"\bhorribl\w*\b",
    r"\babysmal\b",
    r"\bwasted\b",
    r"\bunacceptable\b",
]

_EFFORT_PATTERN = re.compile("|".join(EFFORT_KEYWORDS), flags=re.IGNORECASE)

# ─── Label helpers ────────────────────────────────────────────────────────────

def derive_csat(rating, do_recommend=None) -> int:
    """
    Map reviews.rating (1-5) and optional doRecommend to a 3-class CSAT label.
    """
    try:
        rating = float(rating)
    except (ValueError, TypeError):
        return 1  # fallback neutral

    if rating <= 2:
        return 0  # Dissatisfied
    elif rating == 3:
        rec = str(do_recommend).strip().lower()
        if rec in ("true", "1", "yes"):
            return 2  # Lean Satisfied
        elif rec in ("false", "0", "no"):
            return 0  # Lean Dissatisfied
        return 1  # Neutral
    else:
        return 2  # Satisfied


def derive_ces(text: str) -> int:
    """Return 1 (Difficult) if any effort keyword found, else 0 (Easy)."""
    if not isinstance(text, str):
        return 0
    return 1 if _EFFORT_PATTERN.search(text) else 0


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load 7817_1.csv (Amazon Consumer Reviews) and return a clean labelled
    DataFrame with columns: [combined_text, csat_label, ces_label].
    """
    logger.info(f"Reading: {csv_path}")
    df = pd.read_csv(
        csv_path,
        nrows=max_rows,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )
    logger.info(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")

    # ── Combine title + body ──────────────────────────────────────────────
    title = df[CSV_TITLE_COL].fillna("").astype(str) if CSV_TITLE_COL in df.columns else ""
    body  = df[CSV_TEXT_COL].fillna("").astype(str)
    df["combined_text"] = (title + " " + body).str.strip()

    # ── CSAT label ────────────────────────────────────────────────────────
    has_rec = CSV_RECOMMEND_COL in df.columns
    df["csat_label"] = [
        derive_csat(
            row[CSV_RATING_COL],
            row[CSV_RECOMMEND_COL] if has_rec else None
        )
        for _, row in df.iterrows()
    ]

    # ── CES label ─────────────────────────────────────────────────────────
    logger.info("Scanning text for effort/friction keywords (CES)…")
    df["ces_label"] = df["combined_text"].apply(derive_ces)

    # ── Drop rows without meaningful text ─────────────────────────────────
    df = df[df["combined_text"].str.len() > 15].reset_index(drop=True)

    logger.info(
        f"\nFinal rows : {len(df):,}\n"
        f"CSAT dist  : {df['csat_label'].value_counts().sort_index().to_dict()}\n"
        f"CES  dist  : {df['ces_label'].value_counts().sort_index().to_dict()}"
    )

    return df[["combined_text", "csat_label", "ces_label"]]


# ─── TrustPilot loader ───────────────────────────────────────────────────────

def load_trustpilot(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load TrustPilot reviews CSV and return a labelled DataFrame.
    Columns expected: review_title, review_text, rating
    Returns columns: [combined_text, csat_label, ces_label]
    """
    logger.info(f"Reading TrustPilot: {csv_path}")
    df = pd.read_csv(
        csv_path,
        nrows=max_rows,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )
    logger.info(f"TrustPilot loaded {len(df):,} rows")

    title = df["review_title"].fillna("").astype(str) if "review_title" in df.columns else ""
    body  = df["review_text"].fillna("").astype(str)
    df["combined_text"] = (title + " " + body).str.strip()

    df["csat_label"] = [
        derive_csat(row["rating"]) for _, row in df.iterrows()
    ]
    df["ces_label"] = df["combined_text"].apply(derive_ces)

    df = df[df["combined_text"].str.len() > 15].reset_index(drop=True)
    logger.info(
        f"TrustPilot final rows: {len(df):,}\n"
        f"  CSAT dist: {df['csat_label'].value_counts().sort_index().to_dict()}\n"
        f"  CES  dist: {df['ces_label'].value_counts().sort_index().to_dict()}"
    )
    return df[["combined_text", "csat_label", "ces_label"]]


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    def __init__(self, texts, csat_labels, ces_labels, tokenizer, max_len=256):
        self.texts       = list(texts)
        self.csat_labels = list(csat_labels)
        self.ces_labels  = list(ces_labels)
        self.tokenizer   = tokenizer
        self.max_len     = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        import torch
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "csat_label":     torch.tensor(self.csat_labels[idx], dtype=torch.long),
            "ces_label":      torch.tensor(self.ces_labels[idx],  dtype=torch.long),
        }


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    csv_path: str = "/Users/vedh/Desktop/csat proj  2/7817_1.csv",
    trustpilot_path: str = "/Users/vedh/Desktop/csat proj  2/raw_data/trust_pilot_reviews_data_2022_06.csv",
    model_name: str = "bert-base-uncased",
    max_len: int = 256,
    batch_size: int = 16,
    test_size: float = 0.15,
    val_size: float = 0.15,
    max_rows: int = None,
    num_workers: int = 0,
    random_state: int = 42
):
    """
    Full pipeline: CSV → labels → tokenize → DataLoaders.
    Returns (train_loader, val_loader, test_loader, tokenizer).
    """
    df_amazon = load_dataset(csv_path, max_rows=max_rows)

    # Merge TrustPilot data to fix class imbalance
    import os
    if trustpilot_path and os.path.exists(trustpilot_path):
        df_tp = load_trustpilot(trustpilot_path)
        df = pd.concat([df_amazon, df_tp], ignore_index=True).sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        logger.info(
            f"Combined dataset: {len(df):,} rows\n"
            f"  CSAT dist: {df['csat_label'].value_counts().sort_index().to_dict()}\n"
            f"  CES  dist: {df['ces_label'].value_counts().sort_index().to_dict()}"
        )
    else:
        df = df_amazon

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    texts, csat, ces = df["combined_text"].tolist(), df["csat_label"].tolist(), df["ces_label"].tolist()

    X_train, X_tmp, csat_train, csat_tmp, ces_train, ces_tmp = train_test_split(
        texts, csat, ces,
        test_size=test_size + val_size,
        stratify=csat,
        random_state=random_state
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, csat_val, csat_test, ces_val, ces_test = train_test_split(
        X_tmp, csat_tmp, ces_tmp,
        test_size=1 - rel_val,
        stratify=csat_tmp,
        random_state=random_state
    )
    logger.info(f"Splits → train:{len(X_train):,}  val:{len(X_val):,}  test:{len(X_test):,}")

    def make(t, c, e, shuffle):
        ds = ReviewDataset(t, c, e, tokenizer, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        make(X_train, csat_train, ces_train, True),
        make(X_val,   csat_val,   ces_val,   False),
        make(X_test,  csat_test,  ces_test,  False),
        tokenizer
    )


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, tok = build_dataloaders(
        csv_path="/Users/vedh/Desktop/csat proj  2/7817_1.csv",
        max_rows=3000
    )
    batch = next(iter(train_loader))
    print("\nBatch keys :", list(batch.keys()))
    print("input_ids  :", batch["input_ids"].shape)
    print("csat_labels:", batch["csat_label"][:8].tolist())
    print("ces_labels :", batch["ces_label"][:8].tolist())
