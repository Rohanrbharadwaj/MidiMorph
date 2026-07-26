"""
pretty_midi.synthesize() returns a raw float numpy array, not something
st.audio() can play directly. This converts it to WAV bytes using just the
stdlib `wave` module -- no scipy/soundfile dependency needed for something
this simple.
"""
import io
import wave

import numpy as np
import pretty_midi


def audio_array_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _trim_silence(audio: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Cut any near-zero samples off both ends of a mono audio array.

    Purely amplitude-based (not tied to any particular note timing/envelope
    assumptions), so it works regardless of how the audio was synthesized.
    Returns the array unchanged if it's silent throughout (nothing to trim).
    """
    audible = np.flatnonzero(np.abs(audio) > threshold)
    if audible.size == 0:
        return audio
    return audio[audible[0] : audible[-1] + 1]


def pm_to_wav_bytes(pm: pretty_midi.PrettyMIDI, fs: int = 22050) -> bytes:
    """Render a PrettyMIDI object to playable WAV bytes.

    Uses pm.synthesize() (sine-wave additive synth) rather than fluidsynth --
    good enough for a demo without shipping a soundfont; swap to
    pm.fluidsynth(sf2_path=...) later if you want a richer timbre (see the
    earlier deployment notes about this tradeoff).
    """
    if not pm.instruments or not any(inst.notes for inst in pm.instruments):
        # silent chunk -- return a short buffer of true silence rather than
        # letting synthesize() on an empty instrument raise or return odd length
        return audio_array_to_wav_bytes(np.zeros(fs // 4), fs)
    audio = _trim_silence(pm.synthesize(fs=fs))
    return audio_array_to_wav_bytes(audio, fs)


def pm_to_wav_bytes_and_offset(pm: pretty_midi.PrettyMIDI, fs: int = 22050):
    """Same rendering as pm_to_wav_bytes, but also returns how many seconds
    were trimmed off the FRONT by _trim_silence.

    Needed by anything that overlays visuals synced to the audio's own
    playback clock (e.g. ui/synced_player.py's note highlighting): note
    onset/offset times are computed from the untrimmed MIDI, but
    `audio.currentTime` in the browser runs on the trimmed WAV's clock.
    Without this offset, highlighting drifts by however much leading
    silence got cut -- which varies per chunk depending on where the first
    note actually lands relative to the synth's attack envelope, so it
    can't be assumed to be ~0.

    Returns:
        wav_bytes: same as pm_to_wav_bytes
        offset_seconds: add this to audio.currentTime to get back into the
            original MIDI's time coordinates
    """
    if not pm.instruments or not any(inst.notes for inst in pm.instruments):
        return audio_array_to_wav_bytes(np.zeros(fs // 4), fs), 0.0

    audio = pm.synthesize(fs=fs)
    audible = np.flatnonzero(np.abs(audio) > 1e-3)
    if audible.size == 0:
        return audio_array_to_wav_bytes(audio, fs), 0.0

    trimmed = audio[audible[0] : audible[-1] + 1]
    offset_seconds = float(audible[0]) / fs
    return audio_array_to_wav_bytes(trimmed, fs), offset_seconds
