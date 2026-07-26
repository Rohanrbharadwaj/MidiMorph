"""
Mode 2: pick one of the 6 precomputed tracks, listen to the original 2-bar
chorus, then explore its personalized neighborhood with the 4 MidiMe
super-sliders plus the same 20 PCA sliders from Mode 1.

All personalization is precomputed offline (scripts/precompute_midime.py) --
this file only ever calls midime_model.decode(), never encode() or
train_midime(), so switching tracks is instant.
"""
import random

import streamlit as st

from inference.latent_ops import combined_z
from musicvae.tokenizer import midi_to_token_sequence
from ui.synced_player import render_synced_player
from ui.vertical_sliders import slider_bank

N_PCA_SLIDERS = 20
N_SUPER_SLIDERS = 4
PCA_KEY_PREFIX = "mode2_pca_slider_"
SUPER_KEY_PREFIX = "mode2_super_slider_"
SEED_KEY_PREFIX = "mode2_seed_"


def _fresh_seed() -> int:
    return random.randint(0, 2**31 - 1)


@st.cache_data(show_spinner=False)
def _generate_sample(
    _model,
    _midime_model,
    _pca,
    pca_vals: tuple,
    super_vals: tuple,
    w_base_list: tuple,
    temperature: float,
    seed: int,
):
    import torch

    w_base = torch.tensor(w_base_list, dtype=torch.float32)
    z = combined_z(
        _pca, list(pca_vals), w_base=w_base, midime_model=_midime_model, super_offsets=list(super_vals)
    )
    torch.manual_seed(seed)
    return _model.decoder.sample(z, max_length=32, temperature=temperature)[0].cpu().numpy()


def render(model, midime_model, tracks: dict, pca, chorus_dir: str = "assets/chorus_midis"):
    st.subheader("Personalisation")
    st.caption("Pick a track, hear the original, then explore its personalized latent neighborhood.")

    track_keys = sorted(tracks.keys())
    track_labels = {key: f"Track {i + 1}" for i, key in enumerate(track_keys)}
    track_name = st.selectbox(
        "Track", options=track_keys, format_func=lambda key: track_labels[key]
    )
    track_info = tracks[track_name]

    seed_key = f"{SEED_KEY_PREFIX}{track_name}"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = _fresh_seed()

    original_tokens = midi_to_token_sequence(f"{chorus_dir}/{track_name}.mid")
    st.caption(f"Original · {track_info['bpm']:.0f} BPM")
    render_synced_player(
        original_tokens, bpm=track_info["bpm"], height=220, loop=True, key=f"mode2_original_{track_name}"
    )

    if st.button("Reset Sliders"):
        for i in range(N_SUPER_SLIDERS):
            st.session_state[f"{SUPER_KEY_PREFIX}{i}"] = 0.0
        for i in range(N_PCA_SLIDERS):
            st.session_state[f"{PCA_KEY_PREFIX}{i}"] = 0.0
        st.rerun()

    st.markdown("**Super Sliders**")
    super_vals = slider_bank(
        N_SUPER_SLIDERS,
        SUPER_KEY_PREFIX,
        height=160,
    )

    with st.expander("Fine-grained (20 PCA components)", expanded=True):
        pca_vals = slider_bank(N_PCA_SLIDERS, PCA_KEY_PREFIX, height=120)

    temperature = st.slider("Temperature (higher = more random)", 0.1, 1.5, 0.5, 0.05)

    sample = _generate_sample(
        model,
        midime_model,
        pca,
        tuple(pca_vals),
        tuple(super_vals),
        tuple(track_info["w"].tolist()),
        temperature,
        st.session_state[seed_key],
    )

    st.caption(f"Generated · {track_info['bpm']:.0f} BPM")
    render_synced_player(
        sample, bpm=track_info["bpm"], height=280, loop=True, key=f"mode2_player_{track_name}"
    )
