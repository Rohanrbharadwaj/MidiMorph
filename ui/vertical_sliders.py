"""
Shared vertical-slider bank, used for both the 20 PCA sliders (Mode 1 and
Mode 2) and the 4 MidiMe super sliders (Mode 2). Packs sliders edge-to-edge
in narrow columns and skins them with streamlit-vertical-slider so they read
as a dense row of colored bars -- no value or min/max range text cluttering
each one, just the handle position.

streamlit-vertical-slider is a third-party custom component (pip: 
streamlit-vertical-slider, plus its undeclared runtime dependency 
streamlit-toggle-switch -- both are listed in requirements.txt). It behaves
like any other Streamlit widget: pass `key=...` and, same as st.slider,
setting st.session_state[key] before this function runs (e.g. from a
"Reset Sliders" button) changes what's displayed on the next rerun.
"""
from typing import List, Optional

import streamlit as st
import streamlit_vertical_slider as svs

CYAN = "#50dcff"
MAGENTA = "#ff64d7"
TRACK_COLOR = "rgba(255,255,255,0.08)"
THUMB_COLOR = "#0e0e10"


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """t=0 -> c1, t=1 -> c2. Used to fade each slider's fill color across
    the row, cyan to magenta, echoing the piano roll's pitch gradient."""
    c1 = c1.lstrip("#")
    c2 = c2.lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def slider_bank(
    n: int,
    key_prefix: str,
    min_value: float = -1.0,
    max_value: float = 1.0,
    step: float = 0.1,
    default_value: float = 0.0,
    height: int = 120,
    labels: Optional[List[str]] = None,
) -> List[float]:
    """Renders n vertical sliders packed into n equal-width columns.
    labels: per-slider caption (e.g. "1".."4"); None means no label at all,
    which is what keeps the 20-slider row down to bare colored bars.

    Each vertical_slider() is its own iframe-backed custom component: it
    reports its value back to Python asynchronously, on its own schedule, as
    its frontend boots up. That means on the very first script run after a
    key like this is created (e.g. right after switching modes), the
    component's Python call can return None for a brief instant, before its
    iframe has finished mounting and reported an initial value -- downstream
    code (combined_z -> model.decoder.sample) doesn't tolerate None and either
    throws or produces garbage until a follow-up rerun fixes it, which is what
    reads as "lag" between switching modes and getting real output.
    We pre-seed session_state with default_value before the widget is created
    (so there's always a defined value to fall back on) and fall back to it
    explicitly if the component itself hasn't reported back yet -- so the very
    first render already has n real floats, no waiting on a round trip.
    """
    cols = st.columns(n, gap="small")
    values: List[float] = []
    for i, col in enumerate(cols):
        with col:
            key = f"{key_prefix}{i}"
            if key not in st.session_state:
                st.session_state[key] = default_value
            t = i / max(n - 1, 1)
            val = svs.vertical_slider(
                label=labels[i] if labels else None,
                key=key,
                height=height,
                thumb_shape="square",
                step=step,
                default_value=default_value,
                min_value=min_value,
                max_value=max_value,
                track_color=TRACK_COLOR,
                slider_color=_lerp_hex(CYAN, MAGENTA, t),
                thumb_color=THUMB_COLOR,
                value_always_visible=False,
            )
            if val is None:
                val = st.session_state.get(key, default_value)
            values.append(val)
    return values
