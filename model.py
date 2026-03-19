"""
model.py  –  Multi-Task Transformer for CSAT + CES Prediction
==============================================================
Architecture:
    Shared Encoder  : BERT (bert-base-uncased) or DeBERTa
    Head 1 (CSAT)   : Linear → 3 classes  (Dissatisfied, Neutral, Satisfied)
    Head 2 (CES)    : Linear → 2 classes  (Easy, Difficult)

Both heads share the same Transformer body, learning complementary
representations through multi-task gradient flow.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# ─── Label maps (mirrors data_prep.py) ───────────────────────────────────────

CSAT_CLASSES = ["Dissatisfied", "Neutral", "Satisfied"]   # 3 classes
CES_CLASSES  = ["Easy", "Difficult"]                       # 2 classes


# ─── Model class ─────────────────────────────────────────────────────────────

class MultiTaskReviewModel(nn.Module):
    """
    Dual-head classification model built on top of a pre-trained Transformer.

    Parameters
    ----------
    model_name   : HuggingFace model identifier (default bert-base-uncased).
    num_csat     : Number of CSAT classes  (default 3).
    num_ces      : Number of CES  classes  (default 2).
    dropout      : Dropout probability applied before classification heads.
    freeze_base  : If True, freeze the Transformer and only train the heads.
                   Useful for quick sanity checks.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_csat: int = 3,
        num_ces: int  = 2,
        dropout: float = 0.3,
        freeze_base: bool = False
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size          # 768 for BERT-base

        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # ── CSAT head ────────────────────────────────────────────────────
        self.csat_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_csat)
        )

        # ── CES head ─────────────────────────────────────────────────────
        # CES represents a structurally different signal (effort/friction),
        # so it gets its own independent head capacity.
        self.ces_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_ces)
        )

    def forward(self, input_ids, attention_mask):
        """
        Parameters
        ----------
        input_ids      : (B, L) token ids
        attention_mask : (B, L) attention mask

        Returns
        -------
        csat_logits : (B, num_csat)
        ces_logits  : (B, num_ces)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation as the shared embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]   # (B, hidden)
        cls_emb = self.dropout(cls_emb)

        csat_logits = self.csat_head(cls_emb)           # (B, 3)
        ces_logits  = self.ces_head(cls_emb)            # (B, 2)

        return csat_logits, ces_logits


# ─── Sanity check ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running sanity check on: {device}")

    model = MultiTaskReviewModel().to(device)
    print(model)

    B, L = 4, 64
    dummy_ids  = torch.randint(0, 30522, (B, L)).to(device)
    dummy_mask = torch.ones(B, L, dtype=torch.long).to(device)

    csat_out, ces_out = model(dummy_ids, dummy_mask)
    print(f"\nCSAT logits shape: {csat_out.shape}")   # (4, 3)
    print(f"CES  logits shape: {ces_out.shape}")     # (4, 2)
    print("Model forward pass OK ✓")
