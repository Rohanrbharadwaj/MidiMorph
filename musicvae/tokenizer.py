"""
MIDI <-> token conversion for the melody representation used by MusicVAE/MidiMe.

Event scheme (90-way categorical per timestep):
    0                       -> NO_EVENT   (sustain previous note / silence continues)
    1                       -> NOTE_OFF   (previous note ends)
    2 .. 89 (88 values)     -> NOTE_ON for pitch (MIN_PITCH + index)

This is a monophonic reduction: polyphonic input is collapsed to the highest
active pitch per step. 16 steps/bar, 2 bars/chunk (32 steps) by default.

This file is the single source of truth for tokenization -- previously
duplicated across several notebook cells (midi_to_token_sequence was defined
twice; token_sequence_to_midi and chunks_to_midi each reimplemented their own
close_note logic). Both are consolidated below.
"""
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi

# --- event / pitch constants -------------------------------------------------
NO_EVENT = 0
NOTE_OFF = 1
MIN_PITCH = 21   # A0
MAX_PITCH = 108  # C8
NUM_SPECIAL_EVENTS = 2
OUTPUT_DEPTH = NUM_SPECIAL_EVENTS + (MAX_PITCH - MIN_PITCH + 1)  # 90

# --- timing constants ---------------------------------------------------------
STEPS_PER_QUARTER = 4
STEPS_PER_BAR = 16
BARS_PER_CHUNK = 2
SEQ_LEN = STEPS_PER_BAR * BARS_PER_CHUNK  # 32


# ------------------------------------------------------------------------------
# MIDI -> tokens
# ------------------------------------------------------------------------------

def _get_bpm(pm: pretty_midi.PrettyMIDI) -> float:
    _, tempi = pm.get_tempo_changes()
    return float(tempi[0]) if len(tempi) else 120.0


def _extract_active_pitch_per_step(
    inst: pretty_midi.Instrument,
    num_steps: int,
    seconds_per_step: float,
    time_offset: float = 0.0,
) -> np.ndarray:
    """For each step, the highest pitch active during it (-1 if none)."""
    active_pitch = np.full(num_steps, -1, dtype=np.int64)
    for note in inst.notes:
        start_step = int(round((note.start - time_offset) / seconds_per_step))
        end_step = int(round((note.end - time_offset) / seconds_per_step))
        end_step = max(end_step, start_step + 1)  # every note occupies >=1 step
        for step in range(max(0, start_step), min(num_steps, end_step)):
            if note.pitch > active_pitch[step]:
                active_pitch[step] = note.pitch
    return active_pitch


def _active_pitches_to_tokens(active_pitch: np.ndarray) -> np.ndarray:
    num_steps = len(active_pitch)
    tokens = np.full(num_steps, NO_EVENT, dtype=np.int64)
    prev_pitch = -1
    for step in range(num_steps):
        pitch = active_pitch[step]
        if pitch == -1:
            tokens[step] = NOTE_OFF if prev_pitch != -1 else NO_EVENT
            prev_pitch = -1
        elif pitch == prev_pitch:
            tokens[step] = NO_EVENT
        else:
            pitch_clamped = int(np.clip(pitch, MIN_PITCH, MAX_PITCH))
            tokens[step] = NUM_SPECIAL_EVENTS + (pitch_clamped - MIN_PITCH)
            prev_pitch = pitch
    return tokens


def midi_to_token_sequence(
    midi_path: str, seq_len: int = SEQ_LEN, instrument_index: int = 0
) -> np.ndarray:
    """Tokenize the first `seq_len` steps of one instrument track."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    if not pm.instruments:
        raise ValueError("MIDI file has no instrument tracks.")
    inst = pm.instruments[instrument_index]

    bpm = _get_bpm(pm)
    seconds_per_step = 60.0 / bpm / STEPS_PER_QUARTER

    active_pitch = _extract_active_pitch_per_step(inst, seq_len, seconds_per_step)
    return _active_pitches_to_tokens(active_pitch)


def split_midi_into_chunks(
    midi_path: str,
    bars_per_chunk: int = BARS_PER_CHUNK,
    instrument_index: int = 0,
) -> Tuple[List[np.ndarray], float]:
    """Split a full song into consecutive `bars_per_chunk`-bar token chunks.

    Returns (chunks, bpm) so callers can round-trip back to MIDI at the
    correct tempo without re-parsing the file.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    if not pm.instruments:
        raise ValueError("MIDI file has no instrument tracks.")
    inst = pm.instruments[instrument_index]
    if not inst.notes:
        raise ValueError("Instrument has no notes to split.")

    bpm = _get_bpm(pm)
    seconds_per_step = 60.0 / bpm / STEPS_PER_QUARTER
    steps_per_chunk = bars_per_chunk * STEPS_PER_BAR

    # Deliberately NOT using pm.get_end_time(): a NOTE_OFF token closes the
    # previous note at the START of its step, one step early relative to the
    # sequence's nominal length. If a melody's last event is a note-off
    # landing exactly on a chunk boundary, get_end_time() undercounts by a
    # full step and silently drops the last chunk. Using the last NOTE-ON's
    # start instead avoids this ambiguity.
    last_note_on_step = max(int(round(note.start / seconds_per_step)) for note in inst.notes)
    total_steps = last_note_on_step + 1
    num_chunks = max(1, -(-total_steps // steps_per_chunk))  # ceil div, at least 1

    chunks = []
    for chunk_idx in range(num_chunks):
        time_offset = chunk_idx * steps_per_chunk * seconds_per_step
        active_pitch = _extract_active_pitch_per_step(
            inst, steps_per_chunk, seconds_per_step, time_offset=time_offset
        )
        chunks.append(_active_pitches_to_tokens(active_pitch))
    return chunks, bpm


# ------------------------------------------------------------------------------
# tokens -> MIDI
# ------------------------------------------------------------------------------

def _tokens_to_notes(
    tokens: np.ndarray, seconds_per_step: float, velocity: int, time_offset_steps: int = 0
) -> List[pretty_midi.Note]:
    """Shared note-building logic. Single implementation used by both
    token_sequence_to_midi and chunks_to_midi (previously duplicated)."""
    notes: List[pretty_midi.Note] = []
    current_pitch: Optional[int] = None
    current_start: Optional[int] = None

    def close_note(end_step: int) -> None:
        nonlocal current_pitch, current_start
        if current_pitch is not None:
            notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=current_pitch,
                start=(current_start + time_offset_steps) * seconds_per_step,
                end=(end_step + time_offset_steps) * seconds_per_step,
            ))
        current_pitch = None
        current_start = None

    for step, tok in enumerate(tokens):
        if tok == NO_EVENT:
            continue
        elif tok == NOTE_OFF:
            close_note(step)
        else:
            close_note(step)
            current_pitch = MIN_PITCH + (tok - NUM_SPECIAL_EVENTS)
            current_start = step
    close_note(len(tokens))
    return notes


def token_sequence_to_midi(
    tokens: np.ndarray,
    output_path: Optional[str] = None,
    bpm: float = 120.0,
    velocity: int = 80,
) -> pretty_midi.PrettyMIDI:
    """Render a single token chunk (1D indices or 2D one-hot) to a PrettyMIDI object."""
    tokens = np.asarray(tokens)
    if tokens.ndim == 2:
        tokens = tokens.argmax(axis=-1)

    seconds_per_step = 60.0 / bpm / STEPS_PER_QUARTER
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0)  # acoustic grand piano
    inst.notes.extend(_tokens_to_notes(tokens, seconds_per_step, velocity))
    pm.instruments.append(inst)

    if output_path:
        pm.write(output_path)
    return pm


def chunks_to_midi(
    chunk_list: List[np.ndarray],
    output_path: Optional[str] = None,
    bpm: float = 120.0,
    velocity: int = 80,
    bars_per_chunk: int = BARS_PER_CHUNK,
) -> pretty_midi.PrettyMIDI:
    """Render several consecutive chunks back-to-back into one MIDI file."""
    steps_per_chunk = bars_per_chunk * STEPS_PER_BAR
    seconds_per_step = 60.0 / bpm / STEPS_PER_QUARTER

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0)

    for chunk_idx, tokens in enumerate(chunk_list):
        tokens = np.asarray(tokens)
        if tokens.ndim == 2:
            tokens = tokens.argmax(axis=-1)
        time_offset_steps = chunk_idx * steps_per_chunk
        inst.notes.extend(
            _tokens_to_notes(tokens, seconds_per_step, velocity, time_offset_steps)
        )

    pm.instruments.append(inst)
    if output_path:
        pm.write(output_path)
    return pm


def tokens_to_one_hot(tokens: np.ndarray, output_depth: int = OUTPUT_DEPTH) -> np.ndarray:
    one_hot = np.zeros((len(tokens), output_depth), dtype=np.float32)
    one_hot[np.arange(len(tokens)), tokens] = 1.0
    return one_hot
