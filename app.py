"""
Streamlit entrypoint. Routes between the three modes; owns all the
@st.cache_resource / @st.cache_data wrapping so heavy loads happen once per
server process, not once per slider drag (see inference/loader.py's docstring
for why each of these is cached the way it is).

Expects, relative to this file:
    weights/musicvae_trained.pt   -- your trained MusicVAE (gitignored, local only)
    weights/midime_offline.pt      -- from scripts/precompute_midime.py (committed)
    weights/pca_basis.pt           -- from scripts/precompute_pca.py (committed)
    assets/chorus_midis/*.mid      -- the 6 demo tracks (committed)
"""
import os

import streamlit as st

from inference.loader import load_midime_bundle, load_musicvae_checkpoint, load_pca_bundle
from ui import mode_continuity, mode_midime, mode_random

# Anchored to this file's location, not the shell's cwd -- so `streamlit run app.py`
# works the same whether invoked from the repo root or anywhere else, and matches
# how Streamlit Community Cloud runs it (cwd = repo root, but no need to rely on that).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSICVAE_CHECKPOINT = os.path.join(BASE_DIR, "weights", "musicvae_trained.pt")
MIDIME_BUNDLE = os.path.join(BASE_DIR, "weights", "midime_offline.pt")
PCA_BUNDLE = os.path.join(BASE_DIR, "weights", "pca_basis.pt")
CHORUS_DIR = os.path.join(BASE_DIR, "assets", "chorus_midis")

st.set_page_config(page_title="MidiMorph", page_icon="🎹", layout="wide")


@st.cache_resource
def get_model():
    return load_musicvae_checkpoint(MUSICVAE_CHECKPOINT)


@st.cache_resource
def get_midime_bundle():
    return load_midime_bundle(MIDIME_BUNDLE)


@st.cache_resource
def get_pca_bundle():
    return load_pca_bundle(PCA_BUNDLE)


def main():
    st.title("🎹 MidiMorph")

    try:
        model = get_model()
        midime_model, tracks, w_center = get_midime_bundle()
        pca = get_pca_bundle()
    except FileNotFoundError as e:
        st.error(
            f"Missing a required weights file: {e}\n\n"
            f"Expected:\n"
            f"- `{MUSICVAE_CHECKPOINT}` (your trained checkpoint, kept local -- not in git)\n"
            f"- `{MIDIME_BUNDLE}` (run `scripts/precompute_midime.py`)\n"
            f"- `{PCA_BUNDLE}` (run `scripts/precompute_pca.py`)"
        )
        st.stop()

    modes = ["Latent Continuity", "Random Generation", "MidiMe Personalization"]
    default_mode = st.session_state.get("active_mode", modes[0])
    active_mode = st.sidebar.radio("Mode", modes, index=modes.index(default_mode))
    st.session_state["active_mode"] = active_mode

    if active_mode == "Latent Continuity":
        mode_continuity.render(model, chorus_dir=CHORUS_DIR)
    elif active_mode == "Random Generation":
        mode_random.render(model, pca)
    elif active_mode == "MidiMe Personalization":
        mode_midime.render(model, midime_model, tracks, pca, chorus_dir=CHORUS_DIR)


if __name__ == "__main__":
    main()
