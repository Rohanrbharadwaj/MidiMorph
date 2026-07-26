"""
Replaces the previous split of `st.plotly_chart(...)` + `st.audio(...)`.

Streamlit's native elements are separate, sandboxed pieces of the page --
there's no supported way to attach custom JS that reaches from one native
`<audio>` element into a separate native Plotly chart to highlight notes as
they play. A self-contained `components.html(...)` block sidesteps that:
the SVG piano roll and the `<audio>` tag live in the same iframe, so one
`timeupdate` listener can reach both.

Visual language matches ui/piano_roll.py (same cyan->magenta pitch gradient,
same dark background) and app.py's theme constants, so this doesn't look
like a different widget bolted onto the rest of the page.
"""
import base64

import numpy as np
import streamlit as st

from musicvae.tokenizer import MAX_PITCH, MIN_PITCH, STEPS_PER_BAR, STEPS_PER_QUARTER, token_sequence_to_midi
from ui.audio_utils import pm_to_wav_bytes_and_offset

BG_COLOR = "#0e0e10"
GRID_COLOR = "rgba(255,255,255,0.15)"
TEXT_COLOR = "#e8e8e8"
CYAN = (80, 220, 255)     # low-pitch end -- matches piano_roll.py's _pitch_to_color(t=0)
MAGENTA = (255, 100, 215)  # high-pitch end -- matches _pitch_to_color(t=1)


def _pitch_to_rgb(pitch: int, pmin: int, pmax: int):
    t = 0.5 if pmax == pmin else (pitch - pmin) / (pmax - pmin)
    r = int(CYAN[0] + t * (MAGENTA[0] - CYAN[0]))
    g = int(CYAN[1] + t * (MAGENTA[1] - CYAN[1]))
    b = int(CYAN[2] + t * (MAGENTA[2] - CYAN[2]))
    return r, g, b


def render_synced_player(
    tokens: np.ndarray,
    bpm: float = 120.0,
    height: int = 280,
    loop: bool = True,
    autoplay: bool = False,
    key: str = "synced_player",
):
    """tokens: 1D token indices, or 2D one-hot (will be argmax'd).

    Renders a single component containing both the piano roll and the audio
    player -- notes brighten and gain an outline while they're sounding,
    and a playhead line tracks current position.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim == 2:
        tokens = tokens.argmax(axis=-1)

    seq_len = len(tokens)
    seconds_per_step = 60.0 / bpm / STEPS_PER_QUARTER
    total_seconds = seq_len * seconds_per_step

    pm = token_sequence_to_midi(tokens, bpm=bpm)
    notes = pm.instruments[0].notes if pm.instruments else []
    wav_bytes, offset_seconds = pm_to_wav_bytes_and_offset(pm)
    wav_b64 = base64.b64encode(wav_bytes).decode()

    if notes:
        pitches = [n.pitch for n in notes]
        pmin, pmax = min(pitches), max(pitches)
    else:
        pmin, pmax = MIN_PITCH, MAX_PITCH

    roll_h = max(120, height - 70)  # leave room for the <audio> controls bar
    W = 900

    def x_of(t: float) -> float:
        return (t / total_seconds) * W if total_seconds > 0 else 0.0

    def y_of(pitch: float) -> float:
        # inverted: higher pitch draws higher on screen
        lo, hi = pmin - 2, pmax + 2
        frac = (pitch - lo) / (hi - lo)
        return roll_h - frac * roll_h

    svg = [f'<svg viewBox="0 0 {W} {roll_h}" width="100%" height="{roll_h}" style="display:block;">']
    svg.append(f'<rect x="0" y="0" width="{W}" height="{roll_h}" fill="{BG_COLOR}" />')

    num_bars = seq_len // STEPS_PER_BAR
    for bar in range(num_bars + 1):
        x = x_of(bar * STEPS_PER_BAR * seconds_per_step)
        svg.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{roll_h}" stroke="{GRID_COLOR}" stroke-width="1" />')

    note_h = max(4.0, roll_h / max(1, (pmax - pmin + 4)))
    for note in notes:
        note_end = max(note.end, note.start + seconds_per_step * 0.5)
        x0, x1 = x_of(note.start), x_of(note_end)
        r, g, b = _pitch_to_rgb(note.pitch, pmin, pmax)
        y_center = y_of(note.pitch)
        svg.append(
            f'<rect class="mm-note" data-start="{note.start:.5f}" data-end="{note_end:.5f}" '
            f'x="{x0:.1f}" y="{(y_center - note_h / 2):.1f}" width="{max(2.0, x1 - x0):.1f}" '
            f'height="{note_h:.1f}" rx="2" fill="rgb({r},{g},{b})" opacity="0.55" '
            f'style="transition: opacity 0.05s linear;" />'
        )

    if not notes:
        svg.append(
            f'<text x="{W / 2}" y="{roll_h / 2}" fill="{TEXT_COLOR}" font-size="14" '
            f'text-anchor="middle">(silence)</text>'
        )

    svg.append(
        f'<line id="mm-playhead-{key}" x1="0" y1="0" x2="0" y2="{roll_h}" '
        f'stroke="#ffffff" stroke-width="2" opacity="0" />'
    )
    svg.append("</svg>")
    svg_html = "".join(svg)

    loop_attr = "loop" if loop else ""
    autoplay_attr = "autoplay" if autoplay else ""

    html = f"""
    <div style="background:{BG_COLOR}; border-radius:8px; overflow:hidden;">
      {svg_html}
      <audio id="mm-audio-{key}" controls {loop_attr} {autoplay_attr}
             style="width:100%; display:block; outline:none;">
        <source src="data:audio/wav;base64,{wav_b64}" type="audio/wav">
      </audio>
    </div>
    <script>
    (function() {{
        const audio = document.getElementById('mm-audio-{key}');
        const notes = document.querySelectorAll('.mm-note');
        const playhead = document.getElementById('mm-playhead-{key}');
        const totalSeconds = {total_seconds};
        const offsetSeconds = {offset_seconds};
        const W = {W};

        function update() {{
            // audio.currentTime runs on the TRIMMED wav's clock; note
            // start/end times are in the ORIGINAL (untrimmed) MIDI's time --
            // add the trim offset back to compare apples to apples.
            const t = audio.currentTime + offsetSeconds;

            notes.forEach(function(n) {{
                const start = parseFloat(n.getAttribute('data-start'));
                const end = parseFloat(n.getAttribute('data-end'));
                if (t >= start && t < end) {{
                    n.setAttribute('opacity', '1');
                    n.setAttribute('stroke', '#ffffff');
                    n.setAttribute('stroke-width', '1.5');
                }} else {{
                    n.setAttribute('opacity', '0.55');
                    n.removeAttribute('stroke');
                    n.removeAttribute('stroke-width');
                }}
            }});

            if (totalSeconds > 0) {{
                const x = (t / totalSeconds) * W;
                playhead.setAttribute('x1', x);
                playhead.setAttribute('x2', x);
                playhead.setAttribute('opacity', audio.paused ? '0' : '0.85');
            }}
        }}

        function resetHighlight() {{
            notes.forEach(function(n) {{
                n.setAttribute('opacity', '0.55');
                n.removeAttribute('stroke');
                n.removeAttribute('stroke-width');
            }});
            playhead.setAttribute('opacity', '0');
        }}

        audio.addEventListener('timeupdate', update);
        audio.addEventListener('play', update);
        audio.addEventListener('pause', update);
        audio.addEventListener('seeked', update);
        audio.addEventListener('ended', function() {{
            if (!audio.loop) resetHighlight();
        }});
    }})();
    </script>
    """

    st.iframe(html, height=height + 20)
