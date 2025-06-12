# src/asys_i/components/sae_model.py
"""
Core Philosophy: Separation.
Defines the Sparse Autoencoder PyTorch nn.Module.
High Cohesion: Pure model definition and loss calculation.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from asys_i.orchestration.config_loader import SAEModelConfig

log = logging.getLogger(__name__)

class SparseAutoencoder(nn.Module):
    """
    Tied-weights or Untied Sparse Autoencoder Module.
    Architecture: Input -> Pre-bias -> Encoder -> ReLU -> Decoder -> Post-bias -> Output
    Loss = Reconstruction Loss (MSE) + L1 Sparsity Loss
    """
    def __init__(self, config: SAEModelConfig):
        super().__init__()
        self.config = config
        self.d_in = config.d_in
        self.d_sae = config.d_sae
        self.l1_coefficient = config.l1_coefficient
        
        # Use Parameter directly for flexibility
        self.W_enc = nn.Parameter(torch.randn(self.d_in, self.d_sae) * torch.sqrt(torch.tensor(2.0 / self.d_in)))
        # Initialize decoder weights normalized
        self.W_dec = nn.Parameter(F.normalize(torch.randn(self.d_sae, self.d_in), p=2, dim=0) )

        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        # Initialize decoder bias to handle input distribution offset (important!)
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))
         
        self.relu = nn.ReLU()
        # self.apply(self._init_weights) # Custom init if needed
        log.debug(f"SAE initialized: d_in={self.d_in}, d_sae={self.d_sae}, l1_coeff={self.l1_coefficient}")

    # Geometric median initialization for b_dec can be done externally by Trainer

    def _init_weights(self, module):
         # Example: Xavier/Kaiming init if using nn.Linear
         pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
         """ x: [batch_size, d_in] """
         x_minus_b_dec = x - self.b_dec
         hidden = F.linear(x_minus_b_dec, self.W_enc.T, self.b_enc)
         acts = self.relu(hidden) # Feature activations: [batch_size, d_sae]
         return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
         """ acts: [batch_size, d_sae] """
         # Ensure decoder weights are normalized per feature dim during decode
         # self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0) # Can be slow, do less frequently
         x_reconst = F.linear(acts, self.W_dec.T, self.b_dec) # [batch_size, d_in]
         return x_reconst

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass, calculates reconstruction and losses.
        Returns a dictionary of tensors for loss, reconstruction, acts, etc.
        """
        # Check input dimension
        if x.shape[-1] != self.d_in:
             raise ValueError(f"Input tensor dimension {x.shape[-1]} does not match SAE d_in {self.d_in}")

        # Ghost Grads implementation could go here for dead neuron revival

        acts = self.encode(x)
        x_reconst = self.decode(acts)

        # Losses
        # Reconstruction loss relative to norm of input
        # mse_loss = F.mse_loss(x_reconst, x, reduction='mean')
        recons_loss_per_item = ((x_reconst - x)**2).sum(dim=-1)
        recons_loss = recons_loss_per_item.mean()
        
        # L1 Sparsity loss
        l1_loss = self.l1_coefficient * torch.norm(acts, p=1, dim=-1).mean()
        
        total_loss = recons_loss + l1_loss
        
        # Calculate L0 norm (sparsity: fraction of active neurons)
        l0_norm = (acts > 1e-6).float().sum(dim=-1).mean() # Mean number of active features per item
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
        
    def get_alive_neurons_mask(self, acts: torch.Tensor) -> torch.Tensor:
         """ acts: [batch, d_sae]. Returns mask [d_sae] """
         return (acts.sum(dim=0) > 0)
         
    def save_weights(self, path: str):
         torch.save(self.state_dict(), path)
         
    def load_weights(self, path: str, device: str):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        log.info(f"Loaded SAE weights from {path} to {device}")

