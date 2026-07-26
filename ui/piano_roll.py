"""
Piano-roll rendering for token chunks. Deliberately Plotly rather than
matplotlib: matplotlib means re-encoding to a PNG on every slider drag (a
full Streamlit rerun), which is the main latency trap for "changes in real
time" sliders. A Plotly figure re-renders client-side without that round trip.

No `streamlit` import here -- callers do `st.plotly_chart(render_piano_roll(...))`.
This stays a pure function (tokens in, figure out) so it's testable standalone,
same as tokenizer/model/latent_ops.
"""
import numpy as np
import plotly.graph_objects as go

from musicvae.tokenizer import MAX_PITCH, MIN_PITCH, STEPS_PER_BAR, STEPS_PER_QUARTER, token_sequence_to_midi

# dark theme to match the Glitch/MidiMe aesthetic
BG_COLOR = "#0e0e10"
GRID_COLOR = "rgba(255,255,255,0.15)"
TEXT_COLOR = "#e8e8e8"


def _pitch_to_color(pitch: int, pmin: int, pmax: int) -> str:
    """Cyan (low) -> magenta (high) gradient, normalized to this chunk's range."""
    t = 0.5 if pmax == pmin else (pitch - pmin) / (pmax - pmin)
    r = int(80 + t * 175)
    g = int(220 - t * 120)
    b = int(255 - t * 40)
    return f"rgb({r},{g},{b})"


def render_piano_roll(
    tokens: np.ndarray,
    bpm: float = 120.0,
    height: int = 260,
) -> go.Figure:
    """tokens: 1D token indices, or 2D one-hot (will be argmax'd). Returns a
    ready-to-display Plotly figure -- caller does st.plotly_chart(fig, use_container_width=True).
    """
    tokens = np.asarray(tokens)
    if tokens.ndim == 2:
        tokens = tokens.argmax(axis=-1)

    seq_len = len(tokens)
    seconds_per_step = 60.0 / bpm / STEPS_PER_QUARTER
    total_seconds = seq_len * seconds_per_step

    # reuse the tokenizer's own note-building logic instead of reimplementing
    # NOTE_OFF/pitch handling here -- keeps this file purely visual.
    pm = token_sequence_to_midi(tokens, bpm=bpm)
    notes = pm.instruments[0].notes if pm.instruments else []

    fig = go.Figure()

    if notes:
        pitches = [n.pitch for n in notes]
        pmin, pmax = min(pitches), max(pitches)
    else:
        pmin, pmax = MIN_PITCH, MAX_PITCH

    # bar gridlines every STEPS_PER_BAR steps
    num_bars = seq_len // STEPS_PER_BAR
    for bar in range(num_bars + 1):
        x = bar * STEPS_PER_BAR * seconds_per_step
        fig.add_shape(
            type="line", x0=x, x1=x, y0=pmin - 2, y1=pmax + 2,
            line=dict(width=1, color=GRID_COLOR),
        )

    for note in notes:
        # ensure even very short notes render as a visible sliver
        note_end = max(note.end, note.start + seconds_per_step * 0.5)
        fig.add_shape(
            type="rect",
            x0=note.start, x1=note_end,
            y0=note.pitch - 0.4, y1=note.pitch + 0.4,
            fillcolor=_pitch_to_color(note.pitch, pmin, pmax),
            line=dict(width=0),
        )

    if not notes:
        fig.add_annotation(
            text="(silence)", x=total_seconds / 2, y=(pmin + pmax) / 2,
            showarrow=False, font=dict(color=TEXT_COLOR, size=14),
        )

    fig.update_xaxes(
        range=[0, total_seconds], showgrid=False, zeroline=False,
        title_text="time (s)", color=TEXT_COLOR,
    )
    fig.update_yaxes(
        range=[pmin - 2, pmax + 2], showgrid=False, zeroline=False,
        title_text="pitch (MIDI)", color=TEXT_COLOR,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=10, b=30),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        showlegend=False,
    )
    return fig
