"""
Streamlit entrypoint. Routes between the three modes; owns all the
@st.cache_resource / @st.cache_data wrapping so heavy loads happen once per
server process, not once per slider drag (see inference/loader.py's docstring
for why each of these is cached the way it is).

Also owns the app's visual identity: the piano roll (ui/piano_roll.py) already
maps pitch to a cyan -> magenta gradient on a dark background. Everything here
just extends that same language -- same two colors, same dark surface -- into
the page chrome, so the charts don't look like they're sitting inside a
different, unrelated app.

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

# Same values as ui/piano_roll.py's BG_COLOR / TEXT_COLOR, and the two ends of
# its pitch gradient (_pitch_to_color at t=0 and t=1) -- kept as named
# constants here rather than re-derived, so the two files can't silently drift
# apart. If you ever restyle the piano roll's gradient, update these to match.
BG_COLOR = "#0e0e10"
SURFACE_COLOR = "#17171a"
BORDER_COLOR = "rgba(255,255,255,0.08)"
TEXT_COLOR = "#e8e8e8"
TEXT_MUTED = "#8a8a92"
CYAN = "#50dcff"    # low-pitch end of the piano roll's gradient
MAGENTA = "#ff64d7"  # high-pitch end

st.set_page_config(page_title="MidiMorph", page_icon="🎹", layout="wide")


def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', sans-serif;
        }}

        /* ---- page background ---- */
        .stApp {{
            background: {BG_COLOR};
            color: {TEXT_COLOR};
        }}
        section[data-testid="stSidebar"] {{
            background: {SURFACE_COLOR};
            border-right: 1px solid {BORDER_COLOR};
        }}

        /* ---- constrain content width; wide layout + edge-to-edge text hurts
           readability, keep the wide canvas for the piano rolls/sliders but
           cap prose measure ---- */
        .block-container {{
            max-width: 1200px;
            padding-top: 3.5rem;
        }}

        /* ---- headers in the display face ---- */
        h1, h2, h3 {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            letter-spacing: -0.01em;
        }}

        /* ---- hero title: reuse the piano roll's own pitch gradient as text,
           so the chrome and the charts share one signature ---- */
        .mm-hero-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 2.75rem;
            line-height: 1.3;
            padding-top: 0.15rem;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, {CYAN}, {MAGENTA});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
            z-index: 1;
            overflow: visible;
        }}
        .mm-hero-sub {{
            font-family: 'IBM Plex Sans', sans-serif;
            color: {TEXT_MUTED};
            font-size: 1rem;
            max-width: 640px;
            margin-bottom: 1.75rem;
        }}

        /* ---- sidebar mode switch: restyle the radio into a pill nav ---- */
        section[data-testid="stSidebar"] .stRadio > label {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {TEXT_MUTED};
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
            background: transparent;
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.4rem;
            transition: border-color 0.15s ease, background 0.15s ease;
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
            border-color: {CYAN};
        }}

        /* ---- data readouts (captions, small text) in mono, to tie the UI's
           numbers -- bpm, alpha, slider values -- to the token/sequence
           nature of the thing being controlled ---- */
        [data-testid="stCaptionContainer"], .mm-mono {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {TEXT_MUTED};
        }}

        /* ---- sliders: cyan thumb; no colored/gradiented track fill, no
           boxed value or min/max range readouts -- numbers sit on their own ---- */
        .stSlider [role="slider"] {{
            background-color: {CYAN} !important;
            border-color: {CYAN} !important;
        }}

        [data-testid="stSliderThumbValue"],
        [data-testid="stSliderTickBarMin"],
        [data-testid="stSliderTickBarMax"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            color: {TEXT_COLOR} !important;
        }}

        /* ---- primary buttons: gradient fill matching the hero ---- */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, {CYAN}, {MAGENTA});
            border: none;
            color: #0e0e10;
            font-weight: 600;
        }}
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid {BORDER_COLOR};
        }}
        .stButton > button:hover {{
            border-color: {CYAN};
            color: {CYAN};
        }}

        /* ---- generation output card: a thin gradient border around each
           piano-roll + audio block, echoing the pitch gradient as "this box
           holds pitched content" -- the one place we spend the page's
           boldness beyond the hero ---- */
        .mm-card {{
            border-radius: 12px;
            padding: 1px;
            background: linear-gradient(135deg, {CYAN}55, {MAGENTA}55);
            margin-bottom: 1rem;
        }}
        .mm-card-inner {{
            background: {SURFACE_COLOR};
            border-radius: 11px;
            padding: 1rem 1.25rem 1.25rem;
        }}
        .mm-card-label {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: {TEXT_COLOR};
            text-align: center;
            padding: 0.6rem 0 0.4rem;
        }}

        /* ---- expander / divider tone down to match dark surface ---- */
        .streamlit-expanderHeader {{
            font-family: 'Space Grotesk', sans-serif;
            color: {TEXT_COLOR};
        }}
        hr {{
            border-color: {BORDER_COLOR};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown('<div class="mm-hero-title">MidiMorph</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-hero-sub">A MusicVAE, trained from scratch on POP909.</div>',
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
    # a mode file (e.g. mode_continuity's handoff button) can request a switch
    # by setting this key; consumed here, before the radio owns "active_mode"
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