"""
Mode 1: explore the raw MusicVAE prior via 20 PCA-derived sliders.
No MidiMe, no track selection -- z_center defaults to the prior's mean
(zeros), unless Mode 0 handed off an interpolation endpoint via
st.session_state["starting_z"].
"""
import torch
import streamlit as st

from inference.latent_ops import combined_z
from musicvae.tokenizer import token_sequence_to_midi
from ui.audio_utils import pm_to_wav_bytes
from ui.piano_roll import render_piano_roll

N_SLIDERS = 20
SLIDER_KEY_PREFIX = "mode1_pca_slider_"


def render(model, pca, z_size: int = 256, bpm: float = 120.0):
    st.subheader("Random Generation")
    st.caption("20 sliders, each one a principal component of the trained MusicVAE's latent space.")

    starting_z = st.session_state.pop("starting_z", None)
    w_base = starting_z if starting_z is not None else torch.zeros(z_size)
    if starting_z is not None:
        st.info("Starting from where the continuity demo left off.", icon="🎵")

    if st.button("Reset sliders"):
        for i in range(N_SLIDERS):
            st.session_state[f"{SLIDER_KEY_PREFIX}{i}"] = 0.0
        st.rerun()

    slider_cols = st.columns(4)
    slider_vals = []
    for i in range(N_SLIDERS):
        with slider_cols[i % 4]:
            val = st.slider(
                f"component {i + 1}", -2.0, 2.0, 0.0, 0.1, key=f"{SLIDER_KEY_PREFIX}{i}"
            )
            slider_vals.append(val)

    temperature = st.slider("temperature (higher = more random)", 0.1, 1.5, 0.5, 0.05)

    z = combined_z(pca, slider_vals, w_base=w_base, midime_model=None)
    sample = model.decoder.sample(z, max_length=32, temperature=temperature)[0].cpu().numpy()

    fig = render_piano_roll(sample, bpm=bpm, height=280)
    st.plotly_chart(fig, width='stretch')

    pm = token_sequence_to_midi(sample, bpm=bpm)
    st.audio(pm_to_wav_bytes(pm), format="audio/wav")
