"""
Everything needed to go from "a checkpoint file on disk" to "a ready-to-query
model + fitted PCA basis for the 20 sliders."

Deliberately framework-agnostic (no `streamlit` import here) so it's testable
standalone. In app.py, wrap the two expensive calls in Streamlit's cache
decorators so they only run once per server process, not once per slider drag:

    @st.cache_resource
    def get_model():
        return load_musicvae_checkpoint("weights/musicvae.pt")

    @st.cache_data
    def get_pca(_model, _chunks):          # leading underscore = don't hash arg
        mu_all = encode_dataset(_model, _chunks)
        return fit_pca(mu_all, n_components=20)
"""
import os
import urllib.request
from typing import List, Optional

import numpy as np
import torch

from musicvae.model import MusicVAE
from musicvae.tokenizer import OUTPUT_DEPTH, tokens_to_one_hot


def download_checkpoint(url: str, dest_path: str, force: bool = False) -> str:
    """Fetch a checkpoint from a direct URL (e.g. a GitHub Release asset or a
    Hugging Face `resolve/main/...` link) if it isn't already cached locally.

    For Hugging Face Hub specifically, prefer `huggingface_hub.hf_hub_download`
    instead -- it handles versioning/caching for you. This function is the
    generic fallback for any plain HTTPS URL.
    """
    if os.path.exists(dest_path) and not force:
        return dest_path
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def load_musicvae_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    enc_hidden: int = 512,
    dec_hidden: int = 256,
    z_size: int = 256,
) -> MusicVAE:
    """Instantiate MusicVAE with the architecture the checkpoint was trained
    with, load weights, set eval mode. Raises a clear error if the state_dict
    doesn't match -- usually means enc_hidden/dec_hidden/z_size were changed
    from what the checkpoint expects.
    """
    model = MusicVAE(
        output_depth=OUTPUT_DEPTH, enc_hidden=enc_hidden, dec_hidden=dec_hidden, z_size=z_size
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    # allow either a raw state_dict or a training-checkpoint dict with extra keys
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} doesn't match this MusicVAE's shapes "
            f"(enc_hidden={enc_hidden}, dec_hidden={dec_hidden}, z_size={z_size}). "
            f"Double-check these match what the checkpoint was trained with."
        ) from e

    model.eval()
    return model


@torch.no_grad()
def encode_dataset(
    model: MusicVAE,
    chunks: List[np.ndarray],
    device: str = "cpu",
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode a list of token chunks into their `mu` vectors, batched to avoid
    holding the whole one-hot tensor in memory at once for large datasets.

    Returns: [num_chunks, z_size] tensor of mu's -- this is what PCA gets fit on.
    """
    model.eval()
    all_mu = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        one_hot = np.stack([tokens_to_one_hot(c) for c in batch])
        x = torch.tensor(one_hot, dtype=torch.float32, device=device)
        mu, _ = model.encode(x)
        all_mu.append(mu.cpu())
    return torch.cat(all_mu, dim=0)


def fit_pca(mu_all: torch.Tensor, n_components: int = 20):
    """Fit PCA over the dataset's mu's -- gives the 20 slider directions
    (`pca.components_`) and their scale (`pca.explained_variance_`), both
    consumed directly by `inference.latent_ops.combined_z`.

    Needs num_chunks >= n_components (sklearn requirement for full PCA);
    with only a handful of chunks, lower n_components accordingly.
    """
    from sklearn.decomposition import PCA

    if mu_all.shape[0] < n_components:
        raise ValueError(
            f"Need at least {n_components} chunks to fit {n_components}-component PCA, "
            f"got {mu_all.shape[0]}. Encode more of the dataset, or lower n_components."
        )

    pca = PCA(n_components=n_components)
    pca.fit(mu_all.numpy())
    return pca
