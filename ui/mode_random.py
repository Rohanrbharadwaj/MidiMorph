"""
Mode 1: explore the raw MusicVAE prior via 20 PCA-derived sliders.
No MidiMe, no track selection -- z_center defaults to the prior's mean
(zeros), unless Mode 0 handed off an interpolation endpoint via
st.session_state["starting_z"].

Sampling is autoregressive with torch.multinomial, so decoding the *same* z
twice normally gives two different melodies -- that's fine for "New Melody"
but wrong for "Reset sliders", which should land back on the melody you'd
already heard at center. We pin that down with a per-session torch seed:
sliders alone move deterministically through the seeded sample space, only
"New Melody" reaches for a new seed.
"""
import random

import torch
import streamlit as st

from inference.latent_ops import combined_z
from musicvae.tokenizer import token_sequence_to_midi
from ui.audio_utils import pm_to_wav_bytes
from ui.piano_roll import render_piano_roll
from ui.vertical_sliders import slider_bank

N_SLIDERS = 20
SLIDER_KEY_PREFIX = "mode1_pca_slider_"
SEED_KEY = "mode1_seed"


def _fresh_seed() -> int:
    return random.randint(0, 2**31 - 1)


def render(model, pca, z_size: int = 256, bpm: float = 120.0):
    st.subheader("Random Generation")
    st.caption("20 sliders, each one a principal component of the trained MusicVAE's latent space.")

    if SEED_KEY not in st.session_state:
        st.session_state[SEED_KEY] = _fresh_seed()

    starting_z = st.session_state.pop("starting_z", None)
    w_base = starting_z if starting_z is not None else torch.zeros(z_size)
    if starting_z is not None:
        st.info("Starting from where the continuity demo left off.", icon="🎵")

    reset_col, new_col, _spacer = st.columns([1, 1, 4])
    with reset_col:
        if st.button("Reset Sliders"):
            for i in range(N_SLIDERS):
                st.session_state[f"{SLIDER_KEY_PREFIX}{i}"] = 0.0
            st.rerun()
    with new_col:
        if st.button("New Melody"):
            for i in range(N_SLIDERS):
                st.session_state[f"{SLIDER_KEY_PREFIX}{i}"] = 0.0
            st.session_state[SEED_KEY] = _fresh_seed()
            st.rerun()

    slider_vals = slider_bank(N_SLIDERS, SLIDER_KEY_PREFIX, height=120)

    temperature = st.slider("temperature (higher = more random)", 0.1, 1.5, 0.5, 0.05)

    z = combined_z(pca, slider_vals, w_base=w_base, midime_model=None)
    torch.manual_seed(st.session_state[SEED_KEY])
    sample = model.decoder.sample(z, max_length=32, temperature=temperature)[0].cpu().numpy()

    fig = render_piano_roll(sample, bpm=bpm, height=280)
    st.plotly_chart(fig, width='stretch')

    pm = token_sequence_to_midi(sample, bpm=bpm)
    st.audio(pm_to_wav_bytes(pm), format="audio/wav")
