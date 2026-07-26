"""
Mode 0: proves the latent space is continuous before anyone touches a slider.
Interpolates between two hardcoded chorus tracks, renders every step, and
offers a handoff into Mode 1 starting from where the morph left off.
"""
import streamlit as st

from inference.latent_ops import interpolate_chunks
from musicvae.tokenizer import midi_to_token_sequence
from ui.audio_utils import pm_to_wav_bytes
from ui.piano_roll import render_piano_roll
from musicvae.tokenizer import token_sequence_to_midi

# Furthest-apart pair by repeat-count/character among the 6 extracted hooks --
# see the chorus-extraction notes. Swap these if you find a more dramatic pair.
TRACK_A = "pop909_001_chorus"
TRACK_B = "pop909_200_chorus"


def render(model, chorus_dir: str = "assets/chorus_midis", steps: int = 6, bpm: float = 100.0):
    st.subheader("Latent Space Continuity")
    st.caption(
        f"Morphing from **{TRACK_A}** to **{TRACK_B}** in {steps} steps. "
        "Nearby points in the latent space decode to musically related output -- "
        "that's what makes the sliders in the next two modes explorable rather than random."
    )

    chunk_a = midi_to_token_sequence(f"{chorus_dir}/{TRACK_A}.mid")
    chunk_b = midi_to_token_sequence(f"{chorus_dir}/{TRACK_B}.mid")

    alphas, samples, z_interp = interpolate_chunks(model, chunk_a, chunk_b, steps=steps)

    cols = st.columns(steps)
    for i, col in enumerate(cols):
        with col:
            st.caption(f"α = {alphas[i]:.2f}")
            fig = render_piano_roll(samples[i], bpm=bpm, height=180)
            st.plotly_chart(fig, width='stretch', key=f"continuity_roll_{i}")
            pm = token_sequence_to_midi(samples[i], bpm=bpm)
            st.audio(pm_to_wav_bytes(pm), format="audio/wav")

    st.divider()
    if st.button("Try it yourself, starting from the last step →", type="primary"):
        st.session_state["starting_z"] = z_interp[-1]
        # write to a separate key, not "active_mode" directly -- that key
        # belongs to the sidebar radio widget once it's instantiated, and
        # Streamlit forbids writing to a widget's own key after the fact.
        # app.py checks this key before creating the radio and consumes it there.
        st.session_state["pending_mode_switch"] = "Random Generation"
        st.rerun()
