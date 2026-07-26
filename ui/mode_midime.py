"""
Mode 2: pick one of the 6 precomputed tracks, listen to the original 2-bar
chorus, then explore its personalized neighborhood with the 4 MidiMe
super-sliders plus the same 20 PCA sliders from Mode 1.

All personalization is precomputed offline (scripts/precompute_midime.py) --
this file only ever calls midime_model.decode(), never encode() or
train_midime(), so switching tracks is instant.
"""
import pretty_midi
import streamlit as st

from inference.latent_ops import combined_z
from musicvae.tokenizer import token_sequence_to_midi
from ui.audio_utils import pm_to_wav_bytes
from ui.piano_roll import render_piano_roll

N_PCA_SLIDERS = 20
N_SUPER_SLIDERS = 4
PCA_KEY_PREFIX = "mode2_pca_slider_"
SUPER_KEY_PREFIX = "mode2_super_slider_"


def render(model, midime_model, tracks: dict, pca, chorus_dir: str = "assets/chorus_midis"):
    st.subheader("MidiMe Personalization")
    st.caption("Pick a track, hear the original, then explore its personalized latent neighborhood.")

    track_name = st.selectbox("Track", options=sorted(tracks.keys()))
    track_info = tracks[track_name]

    original_pm = pretty_midi.PrettyMIDI(f"{chorus_dir}/{track_name}.mid")
    st.audio(pm_to_wav_bytes(original_pm), format="audio/wav")
    st.caption(f"↑ original · {track_info['bpm']:.0f} BPM")

    if st.button("Reset sliders"):
        for i in range(N_SUPER_SLIDERS):
            st.session_state[f"{SUPER_KEY_PREFIX}{i}"] = 0.0
        for i in range(N_PCA_SLIDERS):
            st.session_state[f"{PCA_KEY_PREFIX}{i}"] = 0.0
        st.rerun()

    st.markdown("**Personalization sliders**")
    super_cols = st.columns(N_SUPER_SLIDERS)
    super_vals = []
    for i, col in enumerate(super_cols):
        with col:
            val = st.slider(f"trait {i + 1}", -2.0, 2.0, 0.0, 0.1, key=f"{SUPER_KEY_PREFIX}{i}")
            super_vals.append(val)

    with st.expander("Fine-grained (20 PCA components)"):
        pca_cols = st.columns(4)
        pca_vals = []
        for i in range(N_PCA_SLIDERS):
            with pca_cols[i % 4]:
                val = st.slider(
                    f"component {i + 1}", -2.0, 2.0, 0.0, 0.1, key=f"{PCA_KEY_PREFIX}{i}"
                )
                pca_vals.append(val)

    temperature = st.slider("temperature (higher = more random)", 0.1, 1.5, 0.5, 0.05)

    z = combined_z(
        pca, pca_vals, w_base=track_info["w"], midime_model=midime_model, super_offsets=super_vals
    )
    sample = model.decoder.sample(z, max_length=32, temperature=temperature)[0].cpu().numpy()

    fig = render_piano_roll(sample, bpm=track_info["bpm"], height=280)
    st.plotly_chart(fig, width='stretch')

    pm = token_sequence_to_midi(sample, bpm=track_info["bpm"])
    st.audio(pm_to_wav_bytes(pm), format="audio/wav")
