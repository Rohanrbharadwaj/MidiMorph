"""
MidiMe: a small VAE fit on top of a frozen MusicVAE's encoder output.

Given a handful of `mu` vectors from a user's uploaded chunks (already
encoded by the big 256-dim MusicVAE), this learns a compact ~4-dim
distribution over "where in z-space this user's style sits" -- see the
architecture discussion: fitting a full 256-dim density from 20-40 samples
is hopelessly underdetermined, so we deliberately squeeze through a tiny
bottleneck that's tractable with that little data.

The big MusicVAE's weights are never touched. This model only ever sees
`mu` vectors (post-encoder), and only ever hands back reconstructed `z`
vectors for the big decoder to consume -- it has no idea tokens or MIDI
exist.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MidiMe(nn.Module):
    def __init__(self, input_size: int = 256, hidden_size: int = 64, latent_size: int = 4):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.enc_hidden = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_sigma = nn.Linear(hidden_size, latent_size)

        self.dec_hidden = nn.Linear(latent_size, hidden_size)
        self.dec_out = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_hidden(x))
        mu = self.fc_mu(h)
        sigma = F.softplus(self.fc_sigma(h))
        return mu, sigma

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, w: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_hidden(w))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encode(x)
        w = self.reparameterize(mu, sigma)
        x_recon = self.decode(w)
        return x_recon, mu, sigma

    def loss(self, x: torch.Tensor, free_bits: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        x_recon, mu, sigma = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x, reduction="none").sum(dim=-1)

        # free bits applied PER-DIMENSION before summing: with only 4 latent
        # dims, summing first then clamping would let a couple of "free"
        # dimensions offset genuine collapse in the others. Clamping each
        # dimension individually is what actually prevents posterior collapse.
        kl_per_dim = 0.5 * (mu**2 + sigma**2 - torch.log(sigma**2) - 1)
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl = kl_per_dim.sum(dim=-1)

        return recon_loss, kl


def train_midime(
    user_mu: torch.Tensor,
    latent_size: int = 4,
    hidden_size: int = 64,
    epochs: int = 500,
    lr: float = 1e-3,
    free_bits: float = 0.5,
    device: str = "cpu",
) -> MidiMe:
    """Fit a MidiMe model on a user's encoded chunks.

    user_mu: [num_chunks, input_size] -- the frozen big model's `mu` output
    for each of the user's uploaded chunks. Trains full-batch (every step
    sees all chunks) since num_chunks is typically only 20-40 -- there's no
    benefit to minibatching a set that small, and full-batch keeps convergence
    predictable for a live demo.
    """
    input_size = user_mu.shape[-1]
    model = MidiMe(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    user_mu = user_mu.to(device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon_loss, kl = model.loss(user_mu, free_bits=free_bits)
        loss = (recon_loss + kl).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    return model
