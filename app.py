"""
Streamlit entrypoint. Routes between the three modes; owns all the
@st.cache_resource / @st.cache_data wrapping so heavy loads happen once per
server process, not once per slider drag (see inference/loader.py's docstring
for why each of these is cached the way it is).

Visual identity here is pulled directly from the project poster: cream paper
background, near-black ink type, a condensed poster display face for the
headline, and a purple -> blue gradient accent. Piano-roll charts (in
ui/piano_roll.py) stay dark -- they read as an inset "screen" against the
cream page, similar to how the poster's own icon boxes sit as bordered panels
against its paper background.

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSICVAE_CHECKPOINT = os.path.join(BASE_DIR, "weights", "musicvae_trained.pt")
MIDIME_BUNDLE = os.path.join(BASE_DIR, "weights", "midime_offline.pt")
PCA_BUNDLE = os.path.join(BASE_DIR, "weights", "pca_basis.pt")
CHORUS_DIR = os.path.join(BASE_DIR, "assets", "chorus_midis")

# Poster palette -- sampled from the project poster (cream paper, near-black
# ink, purple-to-blue gradient headline). Piano-roll charts keep their own
# dark palette (ui/piano_roll.py) unchanged; these are the page's own colors.
PAPER = "#EDE7DB"
PAPER_RAISED = "#F7F3EA"
INK = "#1A1714"
INK_MUTED = "#6B6459"
BORDER = "rgba(26,23,20,0.14)"
PURPLE = "#6C4FA6"
BLUE = "#3E72D6"

st.set_page_config(page_title="MidiMorph", page_icon="🎹", layout="wide")


def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        /* ---- paper background ---- */
        .stApp {{
            background: {PAPER};
            color: {INK};
        }}
        /* subtle grain, echoing the poster's screen-print texture -- kept
           very low-opacity so it reads as paper, not noise */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.035;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
            z-index: 0;
        }}
        section[data-testid="stSidebar"] {{
            background: {PAPER_RAISED};
            border-right: 1px solid {BORDER};
        }}

        .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
        }}

        h1, h2, h3 {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: {INK};
            letter-spacing: -0.01em;
        }}

        /* ---- hero title: poster-style condensed display face, gradient fill ---- */
        .mm-hero-title {{
            font-family: 'Anton', sans-serif;
            font-weight: 400;
            font-size: 3.5rem;
            letter-spacing: 0.01em;
            line-height: 1;
            margin-bottom: 0.35rem;
            background: linear-gradient(90deg, {INK} 0%, {PURPLE} 60%, {BLUE} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .mm-hero-sub {{
            font-family: 'Inter', sans-serif;
            color: {INK_MUTED};
            font-size: 1.05rem;
            max-width: 620px;
            margin-bottom: 1.75rem;
        }}

        /* ---- sidebar mode switch: bordered poster-panel pills ---- */
        section[data-testid="stSidebar"] .stRadio > label {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {INK_MUTED};
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
            background: {PAPER};
            border: 1.5px solid {BORDER};
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.4rem;
            transition: border-color 0.15s ease;
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
            border-color: {PURPLE};
        }}

        [data-testid="stCaptionContainer"], .mm-mono {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {INK_MUTED};
        }}

        /* ---- sliders: gradient handle/track matching the hero ---- */
        .stSlider [role="slider"] {{
            background-color: {PURPLE} !important;
            border-color: {PURPLE} !important;
        }}
        .stSlider > div > div > div > div {{
            background: linear-gradient(90deg, {PURPLE}, {BLUE}) !important;
        }}

        /* ---- buttons ---- */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, {PURPLE}, {BLUE});
            border: none;
            color: {PAPER};
            font-weight: 600;
        }}
        .stButton > button {{
            border-radius: 6px;
            border: 1.5px solid {BORDER};
            color: {INK};
        }}
        .stButton > button:hover {{
            border-color: {PURPLE};
            color: {PURPLE};
        }}

        /* ---- poster-panel card: bordered box for each generation's output,
           echoing the poster's own bordered icon rows. Dark chart sits inside
           like an inset screen. ---- */
        .mm-card {{
            border: 1.5px solid {INK};
            border-radius: 10px;
            background: {PAPER_RAISED};
            padding: 1.1rem 1.25rem 1.25rem;
            margin-bottom: 1.25rem;
        }}
        .mm-card-label {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {INK_MUTED};
            margin-bottom: 0.5rem;
        }}

        .streamlit-expanderHeader {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: {INK};
        }}
        hr {{
            border-color: {BORDER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown('<div class="mm-hero-title">MidiMorph</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-hero-sub">A from-scratch MusicVAE, trained on POP909, '
        "explored three ways.</div>",
        unsafe_allow_html=True,
    )


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
    inject_theme()
    render_hero()

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
    if "active_mode" not in st.session_state:
        st.session_state["active_mode"] = modes[0]
    pending = st.session_state.pop("pending_mode_switch", None)
    if pending is not None:
        st.session_state["active_mode"] = pending
    st.sidebar.markdown("**Mode**")
    active_mode = st.sidebar.radio("Mode", modes, key="active_mode", label_visibility="collapsed")

    if active_mode == "Latent Continuity":
        mode_continuity.render(model, chorus_dir=CHORUS_DIR)
    elif active_mode == "Random Generation":
        mode_random.render(model, pca)
    elif active_mode == "MidiMe Personalization":
        mode_midime.render(model, midime_model, tracks, pca, chorus_dir=CHORUS_DIR)


if __name__ == "__main__":
    main()
