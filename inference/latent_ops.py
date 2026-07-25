"""
Operations on the MusicVAE latent space, shared across all three app modes:

  Mode 0 (this file's `interpolate_chunks`): hardcoded a->b interpolation,
          proves the latent space is continuous before anyone touches a slider.
  Mode 1: 20 PCA sliders over the raw prior.
  Mode 2: MidiMe's 4 super-sliders + the same 20 PCA sliders, centered on a
          user-selected track's personalized latent region.

`combined_z` is the single shared implementation for slider -> z construction
that Modes 1 and 2 both call (Mode 1 is just Mode 2 with w_base fixed to
w_center and no super-slider offsets) -- see the note in the model.py PR
about not duplicating this logic per-mode.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from musicvae.model import MusicVAE
from musicvae.tokenizer import OUTPUT_DEPTH


@torch.no_grad()
def interpolate_chunks(
    model: MusicVAE,
    chunk_a_tokens: np.ndarray,
    chunk_b_tokens: np.ndarray,
    steps: int = 6,
    temperature: float = 0.5,
    device: str = "cpu",
    travel_fraction: float = 0.5,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Encode two token chunks, linearly interpolate between their mu's, and
    decode each interpolation step. This is the continuity-of-latent-space
    demo: no sliders, no MidiMe -- just proof that nearby points in z decode
    to musically-related output.

    travel_fraction: how far toward mu_b the interpolation actually travels
    (1.0 = alpha=1 lands exactly on mu_b; 0.5 = alpha=1 lands on the midpoint
    of mu_a/mu_b). Deliberately kept < 1.0 by default -- shortening the
    traversed distance makes the morph read as smoother/more continuous,
    at the cost of not fully reaching chunk_b's own character by the last step.

    Returns:
        alphas: [steps] numpy array of interpolation weights, 0.0 -> 1.0
        samples: [steps, seq_len] token indices, one row per interpolation step
    """
    model.eval()

    x_a = F.one_hot(torch.as_tensor(chunk_a_tokens).long(), OUTPUT_DEPTH).float().unsqueeze(0).to(device)
    x_b = F.one_hot(torch.as_tensor(chunk_b_tokens).long(), OUTPUT_DEPTH).float().unsqueeze(0).to(device)

    mu_a, _ = model.encode(x_a)
    mu_b, _ = model.encode(x_b)
    mu_b = mu_a + (mu_b - mu_a) * travel_fraction

    alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)  # [steps, 1]
    z_interp = (1 - alphas) * mu_a + alphas * mu_b  # travels only `travel_fraction` of the way to mu_b

    samples = model.decoder.sample(z_interp, max_length=chunk_a_tokens.shape[-1], temperature=temperature)

    return alphas.squeeze(1).cpu().numpy(), samples.cpu().numpy()


def combined_z(
    pca,
    pca_slider_vals: List[float],
    w_base: torch.Tensor,
    midime_model: Optional["MidiMe"] = None,  # noqa: F821 - see musicvae/midime.py
    super_offsets: Optional[List[float]] = None,
    num_std: float = 2.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Build a single z from slider positions.

    - Mode 1 (random generation): call with midime_model=None, w_base=z_center
      (e.g. torch.zeros(z_size), the prior's mean) and super_offsets=None.
      The PCA delta is added directly to w_base.
    - Mode 2 (MidiMe): call with a trained midime_model, w_base = w_center or
      a specific track's w_mu_real[i], and super_offsets from the 4 sliders.
      w_base + super_offsets is decoded through midime_model into z-space,
      then the PCA delta is added on top.
    """
    stds = np.sqrt(pca.explained_variance_[: len(pca_slider_vals)])
    pca_delta = (np.asarray(pca_slider_vals) * num_std * stds) @ pca.components_
    pca_delta = torch.tensor(pca_delta, dtype=torch.float32, device=device)

    if midime_model is None:
        return (w_base + pca_delta).unsqueeze(0)

    offsets = torch.tensor(super_offsets or [0.0] * w_base.shape[-1], dtype=torch.float32, device=device)
    w = (w_base + offsets).unsqueeze(0)
    with torch.no_grad():
        base_z = midime_model.decode(w)
    return base_z + pca_delta.unsqueeze(0)
