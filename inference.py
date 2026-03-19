"""
inference.py  –  Single-review CSAT + CES prediction
======================================================
Usage (command line):
    python inference.py "The product is great but delivery was late."

Usage (Python):
    from inference import predict
    result = predict("The product is great but delivery was late.")
    print(result)
    # {'csat': 'Satisfied', 'csat_confidence': 0.91,
    #  'ces': 'Difficult',  'ces_confidence': 0.87}
"""

import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path

from model import MultiTaskReviewModel, CSAT_CLASSES, CES_CLASSES


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_CHECKPOINT = "./checkpoints/best_model.pt"
DEFAULT_MAX_LEN    = 256


# ─── Predictor class ─────────────────────────────────────────────────────────

class ReviewPredictor:
    """
    Load a trained MultiTaskReviewModel checkpoint and run inference
    on arbitrary customer review text.

    Example
    -------
    >>> predictor = ReviewPredictor()
    >>> predictor.predict("The product is great but delivery was late.")
    {'csat': 'Satisfied', 'csat_confidence': 0.91, 'ces': 'Difficult', 'ces_confidence': 0.87}
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        model_name: str      = DEFAULT_MODEL_NAME,
        max_len: int         = DEFAULT_MAX_LEN,
        device: str          = None
    ):
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt.get("config", {})
        model_name = cfg.get("model_name", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = MultiTaskReviewModel(model_name=model_name)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        print(f"[ReviewPredictor] Loaded checkpoint from {checkpoint_path} → {self.device}")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Predict CSAT and CES for a single review string.

        Returns
        -------
        dict with keys:
            csat              : str   (e.g. "Satisfied")
            csat_confidence   : float (0-1)
            ces               : str   (e.g. "Difficult")
            ces_confidence    : float (0-1)
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids  = enc["input_ids"].to(self.device)
        attn_mask  = enc["attention_mask"].to(self.device)

        csat_logits, ces_logits = self.model(input_ids, attn_mask)

        csat_probs = F.softmax(csat_logits, dim=-1).squeeze(0)
        ces_probs  = F.softmax(ces_logits,  dim=-1).squeeze(0)

        csat_idx  = csat_probs.argmax().item()
        ces_idx   = ces_probs.argmax().item()

        return {
            "csat"            : CSAT_CLASSES[csat_idx],
            "csat_confidence" : round(csat_probs[csat_idx].item(), 4),
            "ces"             : CES_CLASSES[ces_idx],
            "ces_confidence"  : round(ces_probs[ces_idx].item(), 4),
        }

    def predict_batch(self, texts: list) -> list:
        """Run predict() on a list of review strings."""
        return [self.predict(t) for t in texts]


# ─── Convenience function ─────────────────────────────────────────────────────

_predictor = None  # lazy singleton

def predict(text: str,
            checkpoint: str = DEFAULT_CHECKPOINT,
            model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """
    Module-level predict() — loads the model once and caches it.
    """
    global _predictor
    if _predictor is None:
        _predictor = ReviewPredictor(checkpoint_path=checkpoint,
                                     model_name=model_name)
    return _predictor.predict(text)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    review = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "The product is great but the return process was incredibly frustrating."
    )
    result = predict(review)
    print("\n──────────────────────────────────────────")
    print(f"  Review : {review[:80]}{'…' if len(review)>80 else ''}")
    print(f"  CSAT   : {result['csat']:12s}  confidence={result['csat_confidence']:.2%}")
    print(f"  CES    : {result['ces']:12s}  confidence={result['ces_confidence']:.2%}")
    print("──────────────────────────────────────────")
