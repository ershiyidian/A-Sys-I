# src/asys_i/components/sae_model.py (CONFIRMED FROM YOUR LAST INPUT - GOOD)
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from asys_i.orchestration.config_loader import SAEModelConfig

log = logging.getLogger(__name__)

class SparseAutoencoder(nn.Module):
    def __init__(self, config: SAEModelConfig):
        super().__init__()
        self.config = config
        if not isinstance(config.d_in, int) or config.d_in <= 0:
            raise ValueError(f"SAEModel d_in must be a positive integer at instantiation, got {config.d_in}. It should be resolved from 'auto' by the pipeline before this point.")
        self.d_in = config.d_in
        self.d_sae = config.d_sae
        self.l1_coefficient = config.l1_coefficient

        self.W_enc = nn.Parameter(
            torch.randn(self.d_in, self.d_sae) * torch.sqrt(torch.tensor(2.0 / self.d_in))
        )
        self.W_dec = nn.Parameter(
            F.normalize(torch.randn(self.d_sae, self.d_in), p=2, dim=0)
        )
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))
        self.relu = nn.ReLU()

        log.debug(
            f"SAE initialized: d_in={self.d_in}, d_sae={self.d_sae}, l1_coeff={self.l1_coefficient}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_minus_b_dec = x - self.b_dec
        hidden = F.linear(x_minus_b_dec, self.W_enc.T, self.b_enc)
        acts = self.relu(hidden)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        # W_dec normalization is now handled periodically by the SAETrainerWorker
        x_reconst = F.linear(acts, self.W_dec.T, self.b_dec)
        return x_reconst

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"Input tensor dimension {x.shape[-1]} does not match SAE d_in {self.d_in}"
            )

        acts = self.encode(x)
        x_reconst = self.decode(acts)

        recons_loss_per_item = ((x_reconst - x) ** 2).sum(dim=-1)
        recons_loss = recons_loss_per_item.mean()
        l1_loss = self.l1_coefficient * torch.norm(acts, p=1, dim=-1).mean()
        total_loss = recons_loss + l1_loss
        l0_norm = (acts > 1e-6).float().sum(dim=-1).mean() # Num active features per item
        sparsity_fraction = l0_norm / self.d_sae

        return {
            "loss": total_loss,
            "reconstruction_loss": recons_loss,
            "l1_loss": l1_loss,
            "x_reconstructed": x_reconst,
            "activations": acts,
            "l0_norm": l0_norm,
            "sparsity_fraction": sparsity_fraction,
        }

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: str):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        log.info(f"Loaded SAE weights from {path} to {device}")
