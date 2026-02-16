"""
Session Recording & Transcript Log — Phase 10

Provides:
  - SessionRecorder  : accumulates per-language audio tracks and writes WAVs
  - TranscriptLogger : accumulates structured transcript entries and writes text
  - write_manifest   : generates SHA-256 manifest JSON for all session artifacts

Output directory layout:
    recordings/sessions/{session_name}/
        {session_name}_en.wav
        {session_name}_es.wav
        {session_name}_transcript.txt
        {session_name}_manifest.json
"""

import hashlib
import io
import json
import struct
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Base directory for session recordings
SESSIONS_DIR = Path(__file__).parent.parent / "recordings" / "sessions"


# ------------------------------------------------------------------ #
#  SessionRecorder                                                     #
# ------------------------------------------------------------------ #

class SessionRecorder:
    """
    Accumulates parallel audio tracks — one per language — and writes
    them as WAV files when the session ends.

    Each track interleaves:
      - Original speaker PCM (when someone speaks that language)
      - TTS PCM (translation synthesised *into* that language)
      - Silence padding (to keep both tracks time-aligned)

    All audio is normalised to a uniform sample rate (16 kHz, 16-bit
    mono) regardless of the source.
    """

    SAMPLE_RATE = 16_000   # Uniform output sample rate
    SAMPLE_WIDTH = 2       # 16-bit PCM
    CHANNELS = 1           # Mono

    def __init__(
        self,
        session_name: str,
        language_a: str,
        language_b: str,
        name_a: str = "User A",
        name_b: str = "User B",
    ):
        self.session_name = session_name
        self.language_a = language_a
        self.language_b = language_b
        self.name_a = name_a
        self.name_b = name_b

        # Per-language PCM buffers (list of int16 byte strings)
        self._tracks: dict[str, list[bytes]] = {
            language_a: [],
            language_b: [],
        }

        # Running sample count per track (for alignment)
        self._sample_counts: dict[str, int] = {
            language_a: 0,
            language_b: 0,
        }

        self._start_time = time.time()

    # ------------------------------------------------------------------ #
    #  Audio ingestion                                                     #
    # ------------------------------------------------------------------ #

    def add_speech_pcm(
        self,
        role: str,
        pcm: np.ndarray,
        source_sample_rate: int = 16_000,
    ) -> None:
        """
        Append a speaker's original PCM to their language track and
        pad the other track with equivalent silence for alignment.

        Args:
            role: "a" or "b" — determines which language track.
            pcm:  float32 numpy array in [-1, 1] at *source_sample_rate*.
            source_sample_rate: sample rate of the input PCM.
        """
        lang = self.language_a if role == "a" else self.language_b
        other_lang = self.language_b if role == "a" else self.language_a

        pcm_16k = self._resample_if_needed(pcm, source_sample_rate)
        pcm_bytes = self._float32_to_int16_bytes(pcm_16k)
        n_samples = len(pcm_16k)

        # Append speech to speaker's track
        self._tracks[lang].append(pcm_bytes)
        self._sample_counts[lang] += n_samples

        # Pad other track with silence
        silence = b'\x00\x00' * n_samples  # 2 bytes per int16 sample
        self._tracks[other_lang].append(silence)
        self._sample_counts[other_lang] += n_samples

    def add_tts_pcm(self, target_lang: str, wav_bytes: bytes) -> None:
        """
        Extract raw PCM from TTS WAV output and append to the target
        language track.  The other track is padded with equivalent silence.

        Args:
            target_lang: language code the TTS was synthesised in.
            wav_bytes:   complete WAV file bytes from Piper TTS.
        """
        if not wav_bytes:
            return

        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                tts_sr = wf.getframerate()
                raw_frames = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
        except Exception as e:
            print(f"[SessionRecorder] Failed to parse TTS WAV: {e}")
            return

        # Convert to float32 for resampling
        if sample_width == 2:
            samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            print(f"[SessionRecorder] Unsupported sample width: {sample_width}")
            return

        pcm_16k = self._resample_if_needed(samples, tts_sr)
        pcm_bytes = self._float32_to_int16_bytes(pcm_16k)
        n_samples = len(pcm_16k)

        # Append TTS to target track
        self._tracks[target_lang].append(pcm_bytes)
        self._sample_counts[target_lang] += n_samples

        # Pad the other track with silence
        other_lang = self.language_b if target_lang == self.language_a else self.language_a
        silence = b'\x00\x00' * n_samples
        self._tracks[other_lang].append(silence)
        self._sample_counts[other_lang] += n_samples

    # ------------------------------------------------------------------ #
    #  Finalisation                                                        #
    # ------------------------------------------------------------------ #

    def finalize(self, output_dir: Optional[Path] = None) -> dict[str, Path]:
        """
        Write both language tracks as WAV files.

        Returns:
            dict mapping language code → output WAV path.
        """
        if output_dir is None:
            output_dir = SESSIONS_DIR / self.session_name
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        for lang in [self.language_a, self.language_b]:
            filename = f"{self.session_name}_{lang}.wav"
            wav_path = output_dir / filename
            pcm_data = b"".join(self._tracks[lang])

            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.SAMPLE_WIDTH)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm_data)

            duration_s = self._sample_counts[lang] / self.SAMPLE_RATE
            print(
                f"[SessionRecorder] Wrote {wav_path.name} "
                f"({duration_s:.1f}s, {len(pcm_data)} bytes)"
            )
            paths[lang] = wav_path

        return paths

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resample_if_needed(pcm: np.ndarray, source_sr: int) -> np.ndarray:
        """Simple decimation/interpolation to 16 kHz."""
        target_sr = SessionRecorder.SAMPLE_RATE
        if source_sr == target_sr:
            return pcm
        # Basic linear resampling
        ratio = target_sr / source_sr
        n_out = int(len(pcm) * ratio)
        if n_out == 0:
            return np.array([], dtype=np.float32)
        indices = np.linspace(0, len(pcm) - 1, n_out)
        return np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)

    @staticmethod
    def _float32_to_int16_bytes(pcm: np.ndarray) -> bytes:
        """Convert float32 [-1, 1] PCM to 16-bit little-endian bytes."""
        clipped = np.clip(pcm, -1.0, 1.0)
        int16_data = (clipped * 32767).astype(np.int16)
        return int16_data.tobytes()


# ------------------------------------------------------------------ #
#  TranscriptLogger                                                    #
# ------------------------------------------------------------------ #

class TranscriptLogger:
    """
    Accumulates structured transcript entries during a conversation
    and formats them as a readable turn-by-turn text file.
    """

    def __init__(
        self,
        session_name: str,
        language_a: str,
        language_b: str,
        name_a: str = "User A",
        name_b: str = "User B",
    ):
        self.session_name = session_name
        self.language_a = language_a
        self.language_b = language_b
        self.name_a = name_a
        self.name_b = name_b
        self._entries: list[dict] = []
        self._start_time = time.time()

    def add_entry(
        self,
        role: str,
        text: str,
        language: str,
        translation: Optional[str] = None,
        target_language: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        """
        Record one transcript turn.

        Args:
            role: "a" or "b".
            text: original transcribed text.
            language: source language code.
            translation: translated text (if any).
            target_language: target language code (if any).
            duration: utterance duration in seconds.
        """
        self._entries.append({
            "timestamp_offset": time.time() - self._start_time,
            "role": role,
            "name": self.name_a if role == "a" else self.name_b,
            "text": text,
            "language": language.upper(),
            "translation": translation,
            "target_language": target_language.upper() if target_language else None,
            "duration": duration,
        })

    def format_text(self) -> str:
        """Render the transcript as human-readable turn-by-turn text."""
        total_duration = time.time() - self._start_time
        mins = int(total_duration // 60)
        secs = int(total_duration % 60)

        lines = [
            "SESSION TRANSCRIPT",
            "=" * 60,
            f"Session : {self.session_name}",
            f"Participants : A ({self.name_a}, {self.language_a.upper()}), "
            f"B ({self.name_b}, {self.language_b.upper()})",
            f"Duration: {mins}m {secs:02d}s",
            f"Entries : {len(self._entries)}",
            "=" * 60,
            "",
        ]

        for entry in self._entries:
            offset = entry["timestamp_offset"]
            mm = int(offset // 60)
            ss = int(offset % 60)
            timestamp = f"[{mm:02d}:{ss:02d}]"

            role_label = f"USER {entry['role'].upper()}"
            name = entry["name"]
            lang = entry["language"]

            line = f'{timestamp} {role_label} ({name}, {lang}): "{entry["text"]}"'

            if entry["translation"] and entry["target_language"]:
                line += (
                    f'\n{"":>8}→ ({entry["target_language"]}): '
                    f'"{entry["translation"]}"'
                )

            lines.append(line)
            lines.append("")

        return "\n".join(lines)

    def finalize(self, output_dir: Optional[Path] = None) -> Path:
        """Write transcript to a text file and return the path."""
        if output_dir is None:
            output_dir = SESSIONS_DIR / self.session_name
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.session_name}_transcript.txt"
        txt_path = output_dir / filename
        content = self.format_text()
        txt_path.write_text(content, encoding="utf-8")

        print(
            f"[TranscriptLogger] Wrote {txt_path.name} "
            f"({len(self._entries)} entries, {len(content)} bytes)"
        )
        return txt_path

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ------------------------------------------------------------------ #
#  Manifest writer                                                     #
# ------------------------------------------------------------------ #

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    session_dir: Path,
    session_name: str,
    *,
    start_time: datetime,
    end_time: datetime,
    language_a: str,
    language_b: str,
    name_a: str,
    name_b: str,
    room_id: str,
    transcript_entries: int = 0,
) -> Path:
    """
    Generate a SHA-256 manifest JSON for all session artifacts.

    Hashes every file in *session_dir* (WAVs + transcript) and writes
    the manifest alongside them.

    Returns:
        Path to the manifest JSON file.
    """
    session_dir.mkdir(parents=True, exist_ok=True)

    # Collect all non-manifest files in the session directory
    artifact_files = sorted(
        f for f in session_dir.iterdir()
        if f.is_file() and not f.name.endswith("_manifest.json")
    )

    files_info = []
    for f in artifact_files:
        files_info.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "sha256": _sha256_file(f),
        })

    manifest = {
        "session_name": session_name,
        "room_id": room_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "participants": {
            "a": {"name": name_a, "language": language_a},
            "b": {"name": name_b, "language": language_b},
        },
        "transcript_entries": transcript_entries,
        "files": files_info,
        "generated_at": datetime.now().isoformat(),
    }

    manifest_path = session_dir / f"{session_name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[Manifest] Wrote {manifest_path.name} "
        f"({len(files_info)} files hashed)"
    )
    return manifest_path
