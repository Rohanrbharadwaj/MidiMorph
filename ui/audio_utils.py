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
    audio = pm.synthesize(fs=fs)
    return audio_array_to_wav_bytes(audio, fs)
