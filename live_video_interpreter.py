from __future__ import annotations

import base64
import argparse
import difflib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mss
import numpy as np
import sounddevice as sd
from openai import OpenAI

try:
    import obsws_python as obsws
except ImportError:
    obsws = None

try:
    from vosk import KaldiRecognizer, Model
except ImportError:
    KaldiRecognizer = None
    Model = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def bundled_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    bundled = BUNDLE_DIR / path
    return bundled if bundled.exists() else path


def is_vosk_model_root(path: Path) -> bool:
    return (path / "am").is_dir() and (path / "conf").is_dir() and (path / "graph").is_dir()


def resolve_vosk_model_path(path: Path) -> Path:
    candidates = [path, path / path.name]
    if not path.is_absolute():
        bundled = BUNDLE_DIR / path
        candidates.extend([bundled, bundled / bundled.name])
    for candidate in candidates:
        if is_vosk_model_root(candidate):
            return candidate
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir() and is_vosk_model_root(child):
                return child
    return path


START_REPLAY_BUFFER_COMMANDS = ("start replay buffer",)
STOP_REPLAY_BUFFER_COMMANDS = ("stop replay buffer",)
START_RECORDING_COMMANDS = ("start recording",)
STOP_RECORDING_COMMANDS = ("stop recording",)
OBS_SWITCH_PATTERNS = (
    re.compile(r"\b(?:switch|change|go|cut|transition)\s+(?:to\s+)?(?P<target>.+)$"),
    re.compile(r"\b(?:show|select)\s+(?P<target>.+)$"),
    re.compile(r"\b(?:scene|source)\s+(?P<target>.+)$"),
)


@dataclass(frozen=True)
class VoiceAction:
    kind: str
    target: str | None = None


def extract_obs_switch_target(normalized: str) -> str | None:
    for pattern in OBS_SWITCH_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        target = re.sub(r"\b(?:please|the|scene|source)\b", " ", match.group("target"))
        target = re.sub(r"\s+", " ", target).strip(" .,!?:;")
        if target:
            return target
    return None


def voice_command_action(
    text: str,
    voice_commands: tuple[str, ...],
    enable_obs_scene_source_switching: bool = False,
) -> VoiceAction | None:
    normalized = text.lower().strip()
    if any(command in normalized for command in START_REPLAY_BUFFER_COMMANDS):
        return VoiceAction("start_replay_buffer")
    if any(command in normalized for command in STOP_REPLAY_BUFFER_COMMANDS):
        return VoiceAction("stop_replay_buffer")
    if any(command in normalized for command in START_RECORDING_COMMANDS):
        return VoiceAction("start_recording")
    if any(command in normalized for command in STOP_RECORDING_COMMANDS):
        return VoiceAction("stop_recording")
    if enable_obs_scene_source_switching:
        target = extract_obs_switch_target(normalized)
        if target:
            return VoiceAction("switch_obs_scene_or_source", target)
    if any(command in normalized for command in voice_commands):
        return VoiceAction("clip")
    words = set(re.findall(r"[a-z]+", normalized))
    if words.intersection({"clip", "save", "capture"}) or {"record", "that"}.issubset(words):
        return VoiceAction("clip")
    return None


@dataclass
class OpenAIConfig:
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    voice_command_transcription_model: str = "gpt-4o-mini-transcribe"
    rename_transcription_model: str = "gpt-4o-mini-transcribe"
    naming_model: str = "gpt-4.1-mini"
    max_frames_for_naming: int = 20


@dataclass
class LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    api_key: str | None = None
    api_key_env: str = "LMSTUDIO_API_KEY"
    vision_model: str = "qwen2.5-vl-7b-instruct"


@dataclass
class WhisperLiveConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    api_key_env: str = "WHISPERLIVE_API_KEY"
    model: str = "base.en"
    language: str = "en"


@dataclass
class LocalWhisperConfig:
    model_size: str = "base.en"
    device: str = "auto"
    compute_type: str = "int8"
    cpu_threads: int = 0


@dataclass
class VoiceConfig:
    provider: str = "vosk"
    vosk_model_path: Path = Path("models/vosk-model-en-us-0.22-lgraph")
    rename_vosk_model_path: Path = Path("models/vosk-model-en-us-0.22-lgraph")
    trigger_cooldown_seconds: float = 2.0
    clip_action: str = "obs_replay_buffer"
    enable_obs_scene_source_switching: bool = False
    require_wake_phrase: bool = True
    wake_phrases: tuple[str, ...] = ("clippy", "clip master")
    wake_listen_seconds: float = 8.0


@dataclass
class OBSConfig:
    host: str = "localhost"
    port: int = 4455
    password: str | None = None
    password_env: str = "OBS_WEBSOCKET_PASSWORD"
    request_timeout_seconds: float = 5.0


@dataclass
class AppConfig:
    clip_seconds: int = 45
    segment_seconds: int = 5
    fps: int = 12
    video_source: str = "screen"
    monitor_index: int = 1
    camera_index: int = 0
    camera_width: int | None = None
    camera_height: int | None = None
    video_device_name: str | None = None
    audio_device_name: str | None = None
    audio_device: int | str | None = None
    audio_devices: tuple[int | str | None, ...] = ()
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    command_check_seconds: int = 4
    voice_commands: tuple[str, ...] = (
        "clip that",
        "save that",
        "record that",
        "capture that",
    )
    output_dir: Path = Path("clips")
    buffer_dir: Path = Path(".capture_buffer")
    obs_output_dir: Path | None = None
    obs_clip_extensions: tuple[str, ...] = (".mp4", ".mkv", ".mov", ".flv")
    file_stable_seconds: float = 3.0
    poll_seconds: float = 2.0
    ffmpeg_path: str = "ffmpeg"
    ai_provider: str = "openai"
    voice_command_provider: str = "vosk"
    rename_transcription_provider: str = "local_whisper"
    rename_transcription_audio_fraction: float = 0.5
    name_live_clips: bool = False
    filename_prefix: str = ""
    filename_suffix: str = ""
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    whisperlive: WhisperLiveConfig = field(default_factory=WhisperLiveConfig)
    local_whisper: LocalWhisperConfig = field(default_factory=LocalWhisperConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    obs: OBSConfig = field(default_factory=OBSConfig)


@dataclass
class Segment:
    started_at: float
    ended_at: float
    video_path: Path
    audio_path: Path | None


@dataclass
class AudioRecorder:
    device: int | str | None
    frames: list[np.ndarray]
    stream: sd.InputStream
    input_sample_rate: int


def normalize_audio_device_values(value: Any) -> tuple[int | str | None, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple)):
        raw_values = value
    elif isinstance(value, str) and "," in value:
        raw_values = [part.strip() for part in value.split(",")]
    else:
        raw_values = (value,)

    devices: list[int | str | None] = []
    for raw_value in raw_values:
        if raw_value is None:
            devices.append(None)
            continue
        if raw_value == "":
            continue
        if isinstance(raw_value, int):
            devices.append(raw_value)
            continue
        device = str(raw_value).strip()
        if not device:
            continue
        try:
            devices.append(int(device))
        except ValueError:
            devices.append(device)
    return tuple(devices)


def load_json_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        if cleaned == text:
            raise
        return json.loads(cleaned)


class RollingClipper:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.segments: deque[Segment] = deque()
        self.segment_lock = threading.Lock()
        self.audio_chunks: queue.Queue[Any] = queue.Queue()
        self.stop_event = threading.Event()
        self.voice_transcription_unavailable = False
        self.openai_client = self._build_openai_client()
        self.lmstudio_client = self._build_lmstudio_client()
        self.whisperlive_client = self._build_whisperlive_client()
        self.vosk_model_cache: Model | None = None
        self.local_whisper_model_cache: Any | None = None
        self.local_whisper_model_load_failed = False
        self.voice_awake_until = 0.0
        self.ffmpeg_path = self._resolve_ffmpeg(config.ffmpeg_path)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.trigger_path = self.config.buffer_dir / "clip_now.trigger"

    def _build_openai_client(self) -> OpenAI | None:
        token = self.config.openai.api_key or os.getenv(self.config.openai.api_key_env)
        if not token:
            return None
        return OpenAI(api_key=token)

    def _build_lmstudio_client(self) -> OpenAI | None:
        token = self.config.lmstudio.api_key or os.getenv(self.config.lmstudio.api_key_env) or "lm-studio"
        return OpenAI(base_url=self.config.lmstudio.base_url, api_key=token)

    def _build_whisperlive_client(self) -> OpenAI:
        token = self.config.whisperlive.api_key or os.getenv(self.config.whisperlive.api_key_env) or "whisperlive"
        return OpenAI(base_url=self.config.whisperlive.base_url, api_key=token)

    def run(self) -> None:
        capture_target = self._audio_only_loop if self.config.voice.clip_action == "obs_replay_buffer" else self._capture_loop_safe
        capture_thread = threading.Thread(target=capture_target, name="capture", daemon=True)
        command_thread = threading.Thread(target=self._command_loop, name="voice-commands", daemon=True)
        trigger_thread = threading.Thread(target=self._trigger_loop, name="clip-trigger", daemon=True)
        capture_thread.start()
        command_thread.start()
        trigger_thread.start()
        if self.config.voice.clip_action == "obs_replay_buffer":
            print(
                "OBS voice listener is running. Say 'clip that', 'start replay buffer', "
                "'stop replay buffer', 'start recording', or 'stop recording'. Press Ctrl+C to stop."
            )
        else:
            print(
                "Live Video Interpreter is running. Say 'clip that', 'start replay buffer', "
                "'stop replay buffer', 'start recording', or 'stop recording'. Press Ctrl+C to stop."
            )
        if self.config.voice.require_wake_phrase:
            phrases = " or ".join(f"'{phrase}'" for phrase in self.config.voice.wake_phrases)
            print(
                f"Wake phrase gate is enabled. Say {phrases}, then speak a command within "
                f"{self.config.voice.wake_listen_seconds:g} second(s).",
                flush=True,
            )
        if self.config.voice_command_provider.lower() in {"openai", "whisperlive"}:
            client, _model, label, _language = self._voice_command_transcription_client(
                self.config.voice_command_provider.lower()
            )
            if client is None:
                print(f"Voice command transcription is disabled because {label} is not configured. Use the UI Clip Now button.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop_event.set()
            capture_thread.join(timeout=10)
            command_thread.join(timeout=10)
            trigger_thread.join(timeout=10)

    def _capture_loop_safe(self) -> None:
        try:
            self._capture_loop()
        except Exception as exc:
            print(f"Capture stopped: {exc}")
            self.stop_event.set()

    def _audio_only_loop(self) -> None:
        recorders = self._start_audio_recorders()
        if not recorders:
            print("No audio inputs are available for voice commands.")
            self.stop_event.set()
            return
        devices = ", ".join(str(recorder.device) for recorder in recorders)
        print(f"Listening for voice commands on audio devices: {devices}")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.25)
        finally:
            self._stop_audio_recorders(recorders)

    def _capture_loop(self) -> None:
        source = self.config.video_source.lower()
        if source == "screen":
            self._capture_screen_loop()
            return
        if source == "camera":
            self._capture_camera_loop()
            return
        if source == "directshow":
            self._capture_directshow_loop()
            return
        raise ValueError("video_source must be 'screen', 'camera', or 'directshow'")

    def _capture_screen_loop(self) -> None:
        with mss.mss() as sct:
            monitors = sct.monitors
            if self.config.monitor_index >= len(monitors):
                raise ValueError(f"monitor_index {self.config.monitor_index} is not available")
            monitor = monitors[self.config.monitor_index]
            while not self.stop_event.is_set():
                size = (monitor["width"], monitor["height"])

                def get_frame() -> np.ndarray | None:
                    img = np.array(sct.grab(monitor))
                    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                self._record_segment(get_frame, size)

    def _capture_camera_loop(self) -> None:
        capture = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.config.camera_index}. "
                "Start OBS Virtual Camera and check --list-video-devices."
            )

        if self.config.camera_width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        if self.config.camera_height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        try:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera opened, but no frames were received.")
            height, width = frame.shape[:2]
            size = (width, height)

            def get_frame() -> np.ndarray | None:
                ok, current = capture.read()
                return current if ok else None

            while not self.stop_event.is_set():
                self._record_segment(get_frame, size)
        finally:
            capture.release()

    def _capture_directshow_loop(self) -> None:
        if not self.config.video_device_name:
            raise ValueError("video_device_name is required when video_source is 'directshow'")

        while not self.stop_event.is_set() and not self.voice_transcription_unavailable:
            started = time.time()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = self.config.buffer_dir / f"segment_{stamp}_{int(started)}.mp4"
            audio_path = self.config.buffer_dir / f"segment_{stamp}_{int(started)}.wav"
            input_spec = f"video={self.config.video_device_name}"
            if self.config.audio_device_name:
                input_spec += f":audio={self.config.audio_device_name}"

            command = [
                self.ffmpeg_path,
                "-y",
                "-f",
                "dshow",
                "-rtbufsize",
                "512M",
            ]
            if self.config.camera_width and self.config.camera_height:
                command.extend(["-video_size", f"{self.config.camera_width}x{self.config.camera_height}"])
            input_fps = max(25, self.config.fps)
            command.extend(
                [
                    "-pixel_format",
                    "nv12",
                    "-framerate",
                    str(input_fps),
                    "-i",
                    input_spec,
                    "-t",
                    str(self.config.segment_seconds),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
            if self.config.audio_device_name:
                command.extend(["-c:a", "aac"])
            else:
                command.append("-an")
            command.append(str(video_path))

            recorders = self._start_audio_recorders()

            try:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    details = (result.stderr or result.stdout or "").strip()
                    raise RuntimeError(f"FFmpeg DirectShow capture failed with code {result.returncode}:\n{details}")
            finally:
                self._stop_audio_recorders(recorders)

            ended = time.time()
            mixed_audio = self._mix_audio_recorders(recorders)
            if mixed_audio is not None:
                self._write_wav(audio_path, [mixed_audio])
            elif self.config.audio_device_name:
                try:
                    self._extract_audio(video_path, audio_path)
                except Exception as exc:
                    print(f"Could not extract DirectShow segment audio: {exc}")
                    audio_path = None
            else:
                audio_path = None
            with self.segment_lock:
                self.segments.append(Segment(started, ended, video_path, audio_path))
                self._prune_segments()

    def _record_segment(self, get_frame: Any, size: tuple[int, int]) -> None:
        started = time.time()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.config.buffer_dir / f"segment_{stamp}_{int(started)}.mp4"
        audio_path = self.config.buffer_dir / f"segment_{stamp}_{int(started)}.wav"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, self.config.fps, size)
        frame_interval = 1.0 / max(1, self.config.fps)
        recorders: list[AudioRecorder] = []

        try:
            recorders = self._start_audio_recorders()
            while time.time() - started < self.config.segment_seconds and not self.stop_event.is_set():
                frame_started = time.time()
                frame = get_frame()
                if frame is not None:
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size)
                    writer.write(frame)
                remaining = frame_interval - (time.time() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self._stop_audio_recorders(recorders)
            writer.release()

        ended = time.time()
        mixed_audio = self._mix_audio_recorders(recorders)
        self._write_wav(audio_path, [mixed_audio] if mixed_audio is not None else [])
        with self.segment_lock:
            self.segments.append(Segment(started, ended, video_path, audio_path))
            self._prune_segments()

    def _command_loop(self) -> None:
        provider = self.config.voice_command_provider.lower()
        if provider == "vosk":
            self._vosk_command_loop()
            return
        if provider in {"openai", "whisperlive"}:
            self._api_command_loop(provider)
            return
        if provider == "lmstudio":
            print("LM Studio live voice commands are disabled because LM Studio does not expose compatible live audio transcription in this app. Use Vosk or OpenAI.")
            return
        print(f"Unknown voice command provider: {self.config.voice_command_provider}")

    def _api_command_loop(self, provider: str) -> None:
        client, model, label, language = self._voice_command_transcription_client(provider)
        print(
            f"{label} voice commands enabled with transcription model: {model}. "
            f"Checking recent audio every {self.config.command_check_seconds} second(s).",
            flush=True,
        )
        checks_without_audio = 0
        while not self.stop_event.is_set():
            time.sleep(self.config.command_check_seconds)
            if client is None:
                print(f"{label} transcription client is not configured.", flush=True)
                continue
            chunks = self._drain_recent_audio()
            if not chunks:
                checks_without_audio += 1
                if checks_without_audio == 1 or checks_without_audio % 5 == 0:
                    print(f"{label} voice check: no recent audio chunks available.", flush=True)
                continue
            checks_without_audio = 0
            print(f"{label} voice check: transcribing {len(chunks)} audio chunk(s).", flush=True)
            text = self._transcribe_audio_array(
                chunks,
                prefix="command",
                client=client,
                model=model,
                label=label,
                language=language,
            )
            if self.voice_transcription_unavailable:
                print(
                    f"{label} voice command transcription is unavailable. "
                    "Switch Voice commands to Vosk/OpenAI, or check the local WhisperLive server.",
                    flush=True,
                )
                return
            if not text:
                print(f"{label} voice check: transcription returned no text.", flush=True)
                continue
            normalized = text.lower()
            was_awake = self._voice_is_awake()
            heard_wake_phrase = self._contains_wake_phrase(normalized) if self.config.voice.require_wake_phrase else False
            action = self._voice_command_action_after_wake(normalized)
            if action:
                print(f"Voice command matched: {action.kind} from transcript: {text.strip()}", flush=True)
                try:
                    self.handle_voice_action(action)
                except Exception as exc:
                    print(f"Could not run voice command {action.kind}: {exc}", flush=True)
            elif self.config.voice.require_wake_phrase and not was_awake:
                if not heard_wake_phrase:
                    print(f"{label} speech ignored before wake phrase.", flush=True)
            else:
                print(f"{label} transcript did not match a command: {text.strip()}", flush=True)

    def _vosk_command_loop(self) -> None:
        if Model is None or KaldiRecognizer is None:
            print("Vosk is not installed. Run pip install -r requirements.txt.")
            return
        model_path = self.config.voice.vosk_model_path
        if not model_path.exists():
            print(f"Vosk model not found: {model_path}")
            return

        print(f"Local voice commands enabled with Vosk model: {model_path}")
        try:
            model = Model(str(model_path))
        except Exception as exc:
            print(f"Could not load Vosk model at {model_path}: {exc}")
            return
        recognizers: dict[str, Any] = {}
        last_triggered_at = 0.0
        while not self.stop_event.is_set():
            try:
                item = self.audio_chunks.get(timeout=0.25)
            except queue.Empty:
                continue
            if isinstance(item, tuple) and len(item) == 2:
                device, chunk = item
            else:
                device, chunk = "default", item
            device_key = str(device)
            if device_key not in recognizers:
                recognizers[device_key] = KaldiRecognizer(model, self.config.audio_sample_rate)
            recognizer = recognizers[device_key]
            if chunk.ndim == 2 and chunk.shape[1] > 1:
                chunk = chunk.mean(axis=1, keepdims=True).astype(np.int16)
            accepted = recognizer.AcceptWaveform(chunk.astype(np.int16).tobytes())
            result_text = ""
            if accepted:
                result_text = json.loads(recognizer.Result()).get("text", "")
            else:
                result_text = json.loads(recognizer.PartialResult()).get("partial", "")
            normalized = result_text.lower()
            if not normalized:
                continue
            was_awake = self._voice_is_awake()
            heard_wake_phrase = self._contains_wake_phrase(normalized) if self.config.voice.require_wake_phrase else False
            action = self._voice_command_action_after_wake(normalized)
            if action:
                now = time.time()
                if now - last_triggered_at < self.config.voice.trigger_cooldown_seconds:
                    continue
                last_triggered_at = now
                print(f"Local voice command heard: {normalized}")
                try:
                    self.handle_voice_action(action)
                except Exception as exc:
                    print(f"Could not run voice command: {exc}")
            elif accepted:
                if self.config.voice.require_wake_phrase and not was_awake:
                    if not heard_wake_phrase:
                        print("Vosk speech ignored before wake phrase.", flush=True)
                else:
                    print(f"Vosk transcript did not match a command: {normalized}", flush=True)

    def _is_voice_command(self, text: str) -> bool:
        return self._voice_command_action_after_wake(text) is not None

    def _voice_command_action(self, text: str) -> VoiceAction | None:
        return voice_command_action(
            text,
            self.config.voice_commands,
            self.config.voice.enable_obs_scene_source_switching,
        )

    def _voice_is_awake(self) -> bool:
        return not self.config.voice.require_wake_phrase or time.time() <= self.voice_awake_until

    def _contains_wake_phrase(self, text: str) -> bool:
        normalized = text.lower()
        for phrase in self.config.voice.wake_phrases:
            phrase = phrase.lower().strip()
            if phrase and re.search(rf"\b{re.escape(phrase)}\b", normalized):
                return True
        return False

    def _strip_wake_phrases(self, text: str) -> str:
        cleaned = text.lower()
        for phrase in self.config.voice.wake_phrases:
            phrase = phrase.lower().strip()
            if phrase:
                cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _voice_command_action_after_wake(self, text: str) -> VoiceAction | None:
        normalized = text.lower().strip()
        if not self.config.voice.require_wake_phrase:
            return self._voice_command_action(normalized)

        now = time.time()
        if self._contains_wake_phrase(normalized):
            self.voice_awake_until = now + self.config.voice.wake_listen_seconds
            command_text = self._strip_wake_phrases(normalized)
            print(
                f"Wake phrase heard. Listening for commands for "
                f"{self.config.voice.wake_listen_seconds:g} second(s).",
                flush=True,
            )
            if command_text:
                return self._voice_command_action(command_text)
            return None

        if now <= self.voice_awake_until:
            return self._voice_command_action(normalized)
        return None

    def _extract_obs_switch_target(self, normalized: str) -> str | None:
        return extract_obs_switch_target(normalized)

    def _trigger_loop(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(0.5)
            if not self.trigger_path.exists():
                continue
            self.trigger_path.unlink(missing_ok=True)
            try:
                print("Clip trigger received.")
                self.handle_clip_request()
            except Exception as exc:
                print(f"Could not save triggered clip: {exc}")

    def handle_clip_request(self) -> None:
        if self.config.voice.clip_action == "obs_replay_buffer":
            self.save_obs_replay_buffer()
            return
        clip_path = self.create_clip()
        print(f"Saved clip: {clip_path}")

    def handle_voice_action(self, action: VoiceAction) -> None:
        target = f" -> {action.target}" if action.target else ""
        print(f"Running voice action: {action.kind}{target}", flush=True)
        if action.kind == "clip":
            self.handle_clip_request()
            return
        if action.kind == "start_replay_buffer":
            self.start_obs_replay_buffer()
            return
        if action.kind == "stop_replay_buffer":
            self.stop_obs_replay_buffer()
            return
        if action.kind == "start_recording":
            self.start_obs_recording()
            return
        if action.kind == "stop_recording":
            self.stop_obs_recording()
            return
        if action.kind == "switch_obs_scene_or_source" and action.target:
            self.switch_obs_scene_or_source(action.target)
            return
        raise ValueError(f"Unknown voice action: {action.kind}")

    def _obs_client(self) -> Any:
        if obsws is None:
            raise RuntimeError("obsws-python is not installed. Run pip install -r requirements.txt.")
        password = self.config.obs.password or os.getenv(self.config.obs.password_env, "") or os.getenv("OBS_WEBSOCKET_PASSWORD", "")
        auth_state = "password configured" if password else "no password configured"
        print(
            f"OBS WebSocket connect: {self.config.obs.host}:{self.config.obs.port} "
            f"({auth_state}, env={self.config.obs.password_env})",
            flush=True,
        )
        return obsws.ReqClient(
            host=self.config.obs.host,
            port=self.config.obs.port,
            password=password,
            timeout=self.config.obs.request_timeout_seconds,
        )

    def _obs_value(self, value: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(value, dict) and name in value:
                return value[name]
            if hasattr(value, name):
                return getattr(value, name)
        return default

    def start_obs_replay_buffer(self) -> None:
        client = self._obs_client()
        print("OBS WebSocket command: GetReplayBufferStatus", flush=True)
        status = client.get_replay_buffer_status()
        if getattr(status, "output_active", False):
            print("OBS replay buffer is already running.")
            return
        print("OBS WebSocket command: StartReplayBuffer", flush=True)
        client.start_replay_buffer()
        print("OBS replay buffer start requested.")

    def stop_obs_replay_buffer(self) -> None:
        client = self._obs_client()
        print("OBS WebSocket command: GetReplayBufferStatus", flush=True)
        status = client.get_replay_buffer_status()
        if not getattr(status, "output_active", False):
            print("OBS replay buffer is already stopped.")
            return
        print("OBS WebSocket command: StopReplayBuffer", flush=True)
        client.stop_replay_buffer()
        print("OBS replay buffer stop requested.")

    def start_obs_recording(self) -> None:
        client = self._obs_client()
        print("OBS WebSocket command: GetRecordStatus", flush=True)
        status = client.get_record_status()
        if getattr(status, "output_active", False):
            print("OBS recording is already running.")
            return
        print("OBS WebSocket command: StartRecord", flush=True)
        client.start_record()
        print("OBS recording start requested.")

    def stop_obs_recording(self) -> None:
        client = self._obs_client()
        print("OBS WebSocket command: GetRecordStatus", flush=True)
        status = client.get_record_status()
        if not getattr(status, "output_active", False):
            print("OBS recording is already stopped.")
            return
        print("OBS WebSocket command: StopRecord", flush=True)
        client.stop_record()
        print("OBS recording stop requested.")

    def switch_obs_scene_or_source(self, target: str) -> None:
        client = self._obs_client()
        inventory = self._read_obs_scene_source_inventory(client)
        scene = self._best_obs_name_match(target, inventory["scenes"], "scene")
        if scene:
            print(f"OBS WebSocket command: SetCurrentProgramScene scene={scene}", flush=True)
            client.set_current_program_scene(scene)
            print(f"OBS program scene switched to: {scene}")
            return

        source_names = sorted({item["source_name"] for item in inventory["scene_items"]})
        source = self._best_obs_name_match(target, source_names, "source")
        if not source:
            input_name = self._best_obs_name_match(target, inventory["inputs"], "input")
            if input_name:
                raise RuntimeError(f"OBS input '{input_name}' exists, but it is not present in any scene item.")
            raise RuntimeError(f"No OBS scene or source matched '{target}'.")

        candidates = [item for item in inventory["scene_items"] if item["source_name"] == source]
        if not candidates:
            raise RuntimeError(f"OBS source '{source}' is not present in any scene item.")
        selected = self._choose_scene_item_candidate(candidates, inventory.get("current_scene"))
        scene_name = selected["scene_name"]
        print(f"OBS WebSocket command: SetCurrentProgramScene scene={scene_name}", flush=True)
        client.set_current_program_scene(scene_name)
        if selected.get("scene_item_id") is not None and selected.get("scene_item_enabled") is False:
            print(
                "OBS WebSocket command: SetSceneItemEnabled "
                f"scene={scene_name} item={selected['scene_item_id']} enabled=True",
                flush=True,
            )
            client.set_scene_item_enabled(scene_name, selected["scene_item_id"], True)
        print(f"OBS source selected: {source} in scene {scene_name}")

    def _read_obs_scene_source_inventory(self, client: Any) -> dict[str, Any]:
        print("OBS WebSocket command: GetSceneList", flush=True)
        scene_result = client.get_scene_list()
        scenes_raw = self._obs_value(scene_result, "scenes", default=[])
        current_scene = self._obs_value(scene_result, "current_program_scene_name", "currentProgramSceneName")
        scenes = [self._obs_value(scene, "scene_name", "sceneName") for scene in scenes_raw]
        scenes = [scene for scene in scenes if scene]

        print("OBS WebSocket command: GetInputList", flush=True)
        inputs_result = client.get_input_list()
        inputs_raw = self._obs_value(inputs_result, "inputs", default=[])
        inputs = [self._obs_value(input_item, "input_name", "inputName") for input_item in inputs_raw]
        inputs = [input_name for input_name in inputs if input_name]

        scene_items: list[dict[str, Any]] = []
        for scene_name in scenes:
            try:
                print(f"OBS WebSocket command: GetSceneItemList scene={scene_name}", flush=True)
                items_result = client.get_scene_item_list(scene_name)
            except Exception as exc:
                print(f"Could not read OBS scene items for {scene_name}: {exc}")
                continue
            items_raw = self._obs_value(items_result, "scene_items", "sceneItems", default=[])
            for item in items_raw:
                source_name = self._obs_value(item, "source_name", "sourceName")
                if not source_name:
                    continue
                scene_items.append(
                    {
                        "scene_name": scene_name,
                        "source_name": source_name,
                        "scene_item_id": self._obs_value(item, "scene_item_id", "sceneItemId"),
                        "scene_item_enabled": self._obs_value(item, "scene_item_enabled", "sceneItemEnabled"),
                    }
                )
        return {
            "current_scene": current_scene,
            "scenes": scenes,
            "inputs": inputs,
            "scene_items": scene_items,
        }

    def _choose_scene_item_candidate(self, candidates: list[dict[str, Any]], current_scene: str | None) -> dict[str, Any]:
        if current_scene:
            for candidate in candidates:
                if candidate["scene_name"] == current_scene:
                    return candidate
        return candidates[0]

    def _best_obs_name_match(self, target: str, names: list[str], label: str) -> str | None:
        if not names:
            return None
        target_key = normalize_obs_name(target)
        scored: list[tuple[float, str]] = []
        for name in names:
            name_key = normalize_obs_name(name)
            if not name_key:
                continue
            if name_key == target_key:
                score = 1.0
            elif target_key in name_key or name_key in target_key:
                score = 0.9
            else:
                score = difflib.SequenceMatcher(None, target_key, name_key).ratio()
            scored.append((score, name))
        if not scored:
            return None
        scored.sort(reverse=True)
        best_score, best_name = scored[0]
        if best_score < 0.72:
            return None
        tied = [name for score, name in scored if score >= best_score - 0.04]
        if len(tied) > 1:
            raise RuntimeError(f"Ambiguous OBS {label} '{target}' matched: {', '.join(tied[:5])}.")
        return best_name

    def list_obs_scenes_sources(self) -> None:
        inventory = self._read_obs_scene_source_inventory(self._obs_client())
        print("OBS scenes:")
        for scene in inventory["scenes"]:
            print(f"- {scene}")
        print("OBS sources:")
        for source in sorted({item["source_name"] for item in inventory["scene_items"]}):
            print(f"- {source}")

    def save_obs_replay_buffer(self) -> None:
        client = self._obs_client()
        print("OBS WebSocket command: GetReplayBufferStatus", flush=True)
        status = client.get_replay_buffer_status()
        if not getattr(status, "output_active", False):
            raise RuntimeError("OBS replay buffer is not active. Start Replay Buffer in OBS first.")
        print("OBS WebSocket command: SaveReplayBuffer", flush=True)
        client.save_replay_buffer()
        print("OBS replay buffer save requested.")

    def create_clip(self) -> Path:
        with self.segment_lock:
            cutoff = time.time() - self.config.clip_seconds
            selected = [segment for segment in self.segments if segment.ended_at >= cutoff]

        if not selected:
            raise RuntimeError("No buffered segments are ready yet.")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        draft_path = self.config.output_dir / f"clip_{timestamp}.mp4"
        audio_path = self.config.output_dir / f"clip_{timestamp}.wav"

        combined_video = self._concat_files([segment.video_path for segment in selected], draft_path)
        audio_paths = [segment.audio_path for segment in selected if segment.audio_path]
        if audio_paths:
            combined_audio = self._concat_files(audio_paths, audio_path)
            final_path = self._mux_audio_video(combined_video, combined_audio, draft_path)
        else:
            combined_audio = None
            final_path = combined_video

        if not self.config.name_live_clips:
            if final_path == draft_path:
                return final_path
            default_path = unique_path(draft_path)
            final_path.rename(default_path)
            return default_path

        name = self._suggest_name(final_path, combined_audio)
        safe_name = self._format_clip_filename(name or f"clip_{timestamp}")
        renamed = final_path.with_name(f"{safe_name}.mp4")
        renamed = unique_path(renamed)
        final_path.rename(renamed)

        if combined_audio and combined_audio.exists():
            sidecar = renamed.with_suffix(".wav")
            combined_audio.rename(unique_path(sidecar))
        return renamed

    def watch_obs_folder(self, rename_existing: bool = False) -> None:
        if not self.config.obs_output_dir:
            raise ValueError("obs_output_dir is required for --watch-obs-clips")
        folder = self.config.obs_output_dir
        folder.mkdir(parents=True, exist_ok=True)
        seen = set() if rename_existing else {path.resolve() for path in self._iter_clip_files(folder)}
        print(f"Watching OBS clips in {folder}. Press Ctrl+C to stop.")
        try:
            while True:
                for path in self._iter_clip_files(folder):
                    resolved = path.resolve()
                    if resolved in seen or not self._file_is_stable(path):
                        continue
                    try:
                        renamed = self.rename_clip_file(path)
                        seen.add(renamed.resolve())
                        print(f"Renamed OBS clip: {renamed}")
                    except Exception as exc:
                        print(f"Could not rename {path}: {exc}")
                        seen.add(resolved)
                time.sleep(self.config.poll_seconds)
        except KeyboardInterrupt:
            print("\nStopped watching OBS clips.")

    def batch_rename_obs_clips(self, folder: Path | None = None) -> None:
        folder = folder or self.config.obs_output_dir
        if not folder:
            raise ValueError("obs_output_dir is required for --batch-rename-obs-clips")
        folder.mkdir(parents=True, exist_ok=True)
        paths = self._iter_clip_files(folder)
        if not paths:
            print(f"No OBS clips found in {folder}.", flush=True)
            return
        print(f"Batch scanning {len(paths)} OBS clip(s) in {folder}.", flush=True)
        self._preload_batch_rename_models()
        renamed_count = 0
        skipped_count = 0
        failed_count = 0
        for index, path in enumerate(paths, start=1):
            if not path.exists():
                skipped_count += 1
                continue
            if not self._file_is_settled(path):
                skipped_count += 1
                print(f"[{index}/{len(paths)}] Skipping active or very recent file: {path.name}", flush=True)
                continue
            print(f"[{index}/{len(paths)}] Renaming {path.name}...", flush=True)
            try:
                renamed = self.rename_clip_file(path)
                if renamed != path:
                    renamed_count += 1
                print(f"[{index}/{len(paths)}] Result: {renamed.name}", flush=True)
            except Exception as exc:
                failed_count += 1
                print(f"[{index}/{len(paths)}] Could not rename {path.name}: {exc}", flush=True)
        print(
            "Batch rename complete. "
            f"Renamed {renamed_count}, skipped {skipped_count}, failed {failed_count}, scanned {len(paths)} clip(s).",
            flush=True,
        )

    def _preload_batch_rename_models(self) -> None:
        provider = self.config.rename_transcription_provider.lower()
        if provider in {"local_whisper", "whisper", "faster_whisper"}:
            print("Preloading local Whisper rename transcription model for this batch.", flush=True)
            if self._load_local_whisper_model() is not None:
                print("Local Whisper rename transcription model is loaded and will be reused.", flush=True)

    def rename_clip_file(self, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(path)
        audio_for_naming: Path | None = None
        with tempfile.NamedTemporaryFile(prefix="obs_clip_audio_", suffix=".wav", delete=False) as temp:
            audio_path = Path(temp.name)
        try:
            try:
                self._extract_audio(path, audio_path)
                audio_for_naming = audio_path
            except Exception as exc:
                print(f"Could not extract audio from {path.name}; naming from frames only. {exc}")
            name = self._suggest_name(path, audio_for_naming)
        finally:
            audio_path.unlink(missing_ok=True)
        if not name:
            print("No generated title; leaving filename unchanged.")
            return path
        target = unique_path(path.with_name(f"{self._format_clip_filename(name)}{path.suffix.lower()}"))
        path.rename(target)
        return target

    def _format_clip_filename(self, generated_name: str) -> str:
        parts = [
            sanitize_filename(self.config.filename_prefix, fallback=""),
            sanitize_filename(generated_name),
            sanitize_filename(self.config.filename_suffix, fallback=""),
        ]
        return "_".join(part for part in parts if part) or "clip"

    def _suggest_name(self, video_path: Path, audio_path: Path | None) -> str | None:
        transcript = self._rename_transcript(audio_path)
        frames = sample_video_frames(video_path, self.config.openai.max_frames_for_naming)
        provider = self.config.ai_provider.lower()
        if provider == "lmstudio":
            return self._suggest_name_lmstudio(frames, transcript)
        return self._suggest_name_openai(frames, transcript)

    def _suggest_name_openai(self, frames: list[str], transcript: str) -> str | None:
        if self.openai_client is None:
            print("OPENAI_API_KEY is not set; using timestamp filename.")
            return None

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Name this screen recording clip. Return only a concise filesystem-safe "
                    "title, no extension. Use visible UI/context and transcript. If an RTSS, "
                    "MSI Afterburner, or similar performance overlay is visible, read any clear "
                    "PC specs or benchmark context from it and append compact specs to the title, "
                    "such as GPU model, CPU model, resolution, FPS, or graphics preset. Do not "
                    "invent specs that are not visible. Keep the full filename under 12 words."
                    f"\nTranscript:\n{transcript or '[no clear speech]'}"
                ),
            }
        ]
        for frame in frames:
            content.append({"type": "input_image", "image_url": frame})

        try:
            response = self.openai_client.responses.create(
                model=self.config.openai.naming_model,
                input=[{"role": "user", "content": content}],
            )
            return response.output_text.strip()
        except Exception as exc:
            print(f"OpenAI naming failed: {exc}")
            return None

    def _suggest_name_lmstudio(self, frames: list[str], transcript: str) -> str | None:
        if self.lmstudio_client is None:
            print(f"{self.config.lmstudio.api_key_env} is not set; using timestamp filename.")
            return None
        if not frames:
            print("No sampled frames were available; using timestamp filename.")
            return None

        model_name = self.config.lmstudio.vision_model
        qwen_no_think = "/no_think\n" if "qwen" in model_name.lower() else ""
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    qwen_no_think +
                    "Name this video clip. The first token of your response must be the title. "
                    "Return only one concise filesystem-safe title, no extension, no explanation, "
                    "no markdown, no reasoning. Use visible video context and transcript. If an RTSS, MSI "
                    "Afterburner, or similar performance overlay is visible, read any clear PC "
                    "specs or benchmark context from it and append compact specs to the title, "
                    "such as GPU model, CPU model, resolution, FPS, or graphics preset. Do not "
                    "invent specs that are not visible. Keep the full filename under 12 words."
                    f"\nTranscript:\n{transcript or '[no transcript available]'}"
                ),
            }
        ]
        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": frame}})

        try:
            response = self.lmstudio_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=256,
            )
            choice = response.choices[0]
            title = clean_generated_title(extract_chat_message_text(choice.message))
            if not title:
                title = extract_title_from_reasoning(choice.message)
            if not title:
                finish_reason = getattr(choice, "finish_reason", None)
                print(
                    "LM Studio naming returned no title. "
                    f"finish_reason={finish_reason or 'unknown'} model={model_name}",
                    flush=True,
                )
                return None
            print(f"LM Studio generated title: {title}", flush=True)
            return title
        except Exception as exc:
            print(f"LM Studio naming failed: {exc}")
            return None

    def _voice_command_transcription_client(self, provider: str) -> tuple[OpenAI | None, str, str, str | None]:
        if provider == "whisperlive":
            language = self.config.whisperlive.language.strip() or None
            return self.whisperlive_client, self.config.whisperlive.model, "WhisperLive", language
        return self.openai_client, self.config.openai.voice_command_transcription_model, "OpenAI", None

    def _rename_transcript(self, audio_path: Path | None) -> str:
        if not audio_path:
            return ""
        provider = self.config.rename_transcription_provider.lower()
        if provider in {"", "disabled", "none", "off"}:
            print("Rename audio transcription is disabled; naming from frames only.", flush=True)
            return ""
        if provider in {"local_whisper", "whisper", "faster_whisper"}:
            return self._transcribe_file_local_whisper(audio_path)
        if provider == "vosk":
            return self._transcribe_file_vosk(audio_path)
        if provider != "openai":
            print(f"Unknown rename transcription provider: {self.config.rename_transcription_provider}", flush=True)
            return ""
        if self.openai_client is None:
            print("OpenAI rename transcription is enabled, but OPENAI_API_KEY is not configured.", flush=True)
            return ""
        return self._transcribe_file(
            audio_path,
            client=self.openai_client,
            model=self.config.openai.rename_transcription_model,
            label="OpenAI rename",
        )

    def _transcribe_file(
        self,
        audio_path: Path,
        client: OpenAI | None = None,
        model: str | None = None,
        label: str = "OpenAI",
        language: str | None = None,
    ) -> str:
        selected_client = client or self.openai_client
        selected_model = model or self.config.openai.rename_transcription_model
        if selected_client is None or not audio_path.exists() or audio_path.stat().st_size == 0:
            return ""
        try:
            kwargs: dict[str, Any] = {
                "model": selected_model,
                "file": None,
            }
            if language:
                kwargs["language"] = language
            with audio_path.open("rb") as audio_file:
                kwargs["file"] = audio_file
                result = selected_client.audio.transcriptions.create(
                    **kwargs,
                )
            return getattr(result, "text", "") or ""
        except Exception as exc:
            print(f"{label} transcription failed: {exc}")
            return ""

    def _load_vosk_model(self) -> Any | None:
        if Model is None or KaldiRecognizer is None:
            print("Vosk is not installed. Run pip install -r requirements.txt.", flush=True)
            return None
        if self.vosk_model_cache is not None:
            return self.vosk_model_cache
        model_path = self.config.voice.rename_vosk_model_path
        if not model_path.exists():
            print(f"Vosk model not found: {model_path}", flush=True)
            return None
        try:
            print(f"Loading Vosk rename transcription model: {model_path}", flush=True)
            self.vosk_model_cache = Model(str(model_path))
            return self.vosk_model_cache
        except Exception as exc:
            print(f"Could not load Vosk model at {model_path}: {exc}", flush=True)
            return None

    def _transcribe_file_vosk(self, audio_path: Path) -> str:
        model = self._load_vosk_model()
        if model is None or not audio_path.exists() or audio_path.stat().st_size == 0:
            return ""
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                if sample_width != 2:
                    print(
                        f"Vosk rename transcription expected 16-bit PCM WAV, got {sample_width * 8}-bit audio.",
                        flush=True,
                    )
                    return ""
                recognizer = KaldiRecognizer(model, sample_rate)
                while True:
                    data = wav_file.readframes(4000)
                    if not data:
                        break
                    if channels > 1:
                        audio = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
                        data = audio.mean(axis=1).astype(np.int16).tobytes()
                    recognizer.AcceptWaveform(data)
                result = json.loads(recognizer.FinalResult())
                transcript = str(result.get("text", "")).strip()
                if transcript:
                    print(f"Vosk rename transcript: {transcript}", flush=True)
                else:
                    print("Vosk rename transcription returned no text.", flush=True)
                return transcript
        except Exception as exc:
            print(f"Vosk rename transcription failed: {exc}", flush=True)
            return ""

    def _load_local_whisper_model(self) -> Any | None:
        if WhisperModel is None:
            print(
                "Local Whisper rename transcription requires faster-whisper. "
                "Run pip install -r requirements.txt.",
                flush=True,
            )
            return None
        if self.local_whisper_model_cache is not None:
            return self.local_whisper_model_cache
        if self.local_whisper_model_load_failed:
            return None
        config = self.config.local_whisper
        model_size_or_path = str(config.model_size or "base.en")
        kwargs: dict[str, Any] = {
            "device": config.device or "auto",
            "compute_type": config.compute_type or "int8",
        }
        if config.cpu_threads > 0:
            kwargs["cpu_threads"] = config.cpu_threads
        try:
            print(
                "Loading local Whisper rename transcription model: "
                f"{model_size_or_path} ({kwargs['device']}, {kwargs['compute_type']})",
                flush=True,
            )
            self.local_whisper_model_cache = WhisperModel(model_size_or_path, **kwargs)
            return self.local_whisper_model_cache
        except Exception as exc:
            self.local_whisper_model_load_failed = True
            print(f"Could not load local Whisper model {model_size_or_path}: {exc}", flush=True)
            return None

    def _transcribe_file_local_whisper(self, audio_path: Path) -> str:
        model = self._load_local_whisper_model()
        if model is None or not audio_path.exists() or audio_path.stat().st_size == 0:
            return ""
        try:
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=1,
                vad_filter=True,
                word_timestamps=False,
            )
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
            if transcript:
                language = getattr(info, "language", "unknown")
                probability = getattr(info, "language_probability", 0.0)
                print(
                    f"Local Whisper rename transcript ({language}, {probability:.2f}): {transcript}",
                    flush=True,
                )
            else:
                print("Local Whisper rename transcription returned no text.", flush=True)
            return transcript
        except Exception as exc:
            print(f"Local Whisper rename transcription failed: {exc}", flush=True)
            return ""

    def _transcribe_audio_array(
        self,
        chunks: list[np.ndarray],
        prefix: str,
        client: OpenAI | None = None,
        model: str | None = None,
        label: str = "OpenAI",
        language: str | None = None,
    ) -> str:
        selected_client = client or self.openai_client
        if selected_client is None:
            return ""
        with tempfile.NamedTemporaryFile(prefix=f"{prefix}_", suffix=".wav", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            self._write_wav(temp_path, chunks)
            return self._transcribe_file(
                temp_path,
                client=selected_client,
                model=model,
                label=label,
                language=language,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _drain_recent_audio(self) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        while True:
            try:
                item = self.audio_chunks.get_nowait()
                chunk = item[1] if isinstance(item, tuple) and len(item) == 2 else item
                if isinstance(chunk, np.ndarray):
                    chunks.append(chunk)
            except queue.Empty:
                break
        return chunks

    def _concat_files(self, paths: list[Path], output_path: Path) -> Path:
        if len(paths) == 1:
            shutil.copy2(paths[0], output_path)
            return output_path

        list_path = output_path.with_suffix(".txt")
        list_path.write_text(
            "".join(f"file '{path.resolve().as_posix()}'\n" for path in paths),
            encoding="utf-8",
        )
        command = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return output_path
        finally:
            list_path.unlink(missing_ok=True)

    def _mux_audio_video(self, video_path: Path, audio_path: Path, draft_path: Path) -> Path:
        muxed_path = draft_path.with_name(f"{draft_path.stem}_with_audio.mp4")
        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(muxed_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        video_path.unlink(missing_ok=True)
        return muxed_path

    def _extract_audio(self, video_path: Path, audio_path: Path) -> Path:
        duration_limit = self._rename_audio_duration_limit_seconds(video_path)
        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.config.audio_sample_rate),
        ]
        if duration_limit:
            command.extend(["-t", f"{duration_limit:.3f}"])
        command.append(str(audio_path))
        subprocess.run(command, check=True, capture_output=True, text=True)
        if duration_limit:
            print(
                f"Extracted first {self.config.rename_transcription_audio_fraction:.0%} "
                f"of clip audio for rename transcription ({duration_limit:.1f}s).",
                flush=True,
            )
        return audio_path

    def _rename_audio_duration_limit_seconds(self, video_path: Path) -> float | None:
        fraction = self.config.rename_transcription_audio_fraction
        if fraction <= 0 or fraction >= 1:
            return None
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None
        try:
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        finally:
            capture.release()
        if frame_count <= 0 or fps <= 0:
            return None
        return max(1.0, (frame_count / fps) * fraction)

    def _prune_segments(self) -> None:
        keep_after = time.time() - (self.config.clip_seconds + self.config.segment_seconds * 3)
        while self.segments and self.segments[0].ended_at < keep_after:
            old = self.segments.popleft()
            old.video_path.unlink(missing_ok=True)
            if old.audio_path:
                old.audio_path.unlink(missing_ok=True)

    def _write_wav(self, path: Path, chunks: list[np.ndarray]) -> None:
        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.config.audio_channels), dtype=np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(self.config.audio_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config.audio_sample_rate)
            wav_file.writeframes(audio.astype(np.int16).tobytes())

    def _configured_audio_devices(self) -> tuple[int | str | None, ...]:
        if self.config.audio_devices:
            return self.config.audio_devices
        return (self.config.audio_device,)

    def _device_default_sample_rate(self, device: int | str | None) -> int:
        try:
            info = sd.query_devices(device, "input")
            default_rate = int(float(info.get("default_samplerate", self.config.audio_sample_rate)))
            return default_rate or self.config.audio_sample_rate
        except Exception:
            return self.config.audio_sample_rate

    def _resample_audio_chunk(self, chunk: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == self.config.audio_sample_rate or chunk.size == 0:
            return chunk.astype(np.int16)
        source_length = chunk.shape[0]
        target_length = max(1, int(round(source_length * self.config.audio_sample_rate / source_rate)))
        source_positions = np.linspace(0.0, 1.0, num=source_length, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        audio = chunk.astype(np.float32)
        if audio.ndim == 1:
            resampled = np.interp(target_positions, source_positions, audio)
            return np.clip(resampled, -32768, 32767).astype(np.int16)
        channels = [
            np.interp(target_positions, source_positions, audio[:, channel])
            for channel in range(audio.shape[1])
        ]
        return np.clip(np.stack(channels, axis=1), -32768, 32767).astype(np.int16)

    def _open_audio_stream(
        self,
        device: int | str | None,
        callback: Any,
    ) -> tuple[sd.InputStream, int]:
        sample_rates = [self.config.audio_sample_rate]
        default_rate = self._device_default_sample_rate(device)
        if default_rate not in sample_rates:
            sample_rates.append(default_rate)
        last_error: Exception | None = None
        for sample_rate in sample_rates:
            try:
                stream = sd.InputStream(
                    samplerate=sample_rate,
                    channels=self.config.audio_channels,
                    device=device,
                    dtype="int16",
                    callback=callback,
                )
                return stream, sample_rate
            except Exception as exc:
                last_error = exc
                if sample_rate == self.config.audio_sample_rate:
                    print(f"Audio input {device} rejected {sample_rate} Hz; trying default device rate {default_rate} Hz.")
        raise last_error or RuntimeError(f"Could not open audio input {device}")

    def _start_audio_recorders(self) -> list[AudioRecorder]:
        recorders: list[AudioRecorder] = []
        devices = self._configured_audio_devices()
        device_labels = ["system default" if device is None else str(device) for device in devices]
        print(f"Configured audio input device(s): {', '.join(device_labels)}", flush=True)
        for device in devices:
            frames: list[np.ndarray] = []
            input_sample_rate = self._device_default_sample_rate(device)

            def audio_callback(
                indata: np.ndarray,
                frame_count: int,
                time_info: Any,
                status: sd.CallbackFlags,
                captured_frames: list[np.ndarray] = frames,
                captured_device: int | str | None = device,
            ) -> None:
                if status:
                    print(f"Audio warning on {captured_device}: {status}")
                chunk = self._resample_audio_chunk(indata.copy(), input_sample_rate)
                captured_frames.append(chunk)
                self.audio_chunks.put((captured_device, chunk))

            try:
                stream, opened_sample_rate = self._open_audio_stream(device, audio_callback)
                input_sample_rate = opened_sample_rate
                stream.start()
                opened_label = "system default" if device is None else str(device)
                print(f"Audio input {opened_label} opened at {opened_sample_rate} Hz.", flush=True)
                if opened_sample_rate != self.config.audio_sample_rate:
                    print(
                        f"Audio input {opened_label} is being resampled to {self.config.audio_sample_rate} Hz."
                    )
                recorders.append(
                    AudioRecorder(
                        device=device,
                        frames=frames,
                        stream=stream,
                        input_sample_rate=opened_sample_rate,
                    )
                )
            except Exception as exc:
                print(f"Audio input unavailable for {device}: {exc}")
        return recorders

    def _stop_audio_recorders(self, recorders: list[AudioRecorder]) -> None:
        for recorder in recorders:
            recorder.stream.stop()
            recorder.stream.close()

    def _mix_audio_recorders(self, recorders: list[AudioRecorder]) -> np.ndarray | None:
        tracks: list[np.ndarray] = []
        for recorder in recorders:
            if not recorder.frames:
                continue
            audio = np.concatenate(recorder.frames, axis=0).astype(np.float32)
            if audio.ndim == 2 and audio.shape[1] > 1:
                audio = audio.mean(axis=1, keepdims=True)
            elif audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            tracks.append(audio)
        if not tracks:
            return None
        max_length = max(track.shape[0] for track in tracks)
        mixed = np.zeros((max_length, 1), dtype=np.float32)
        for track in tracks:
            mixed[: track.shape[0], 0] += track[:, 0]
        mixed /= max(1, len(tracks))
        return np.clip(mixed, -32768, 32767).astype(np.int16)

    def _resolve_ffmpeg(self, configured: str) -> str:
        if shutil.which(configured):
            return configured
        configured_path = Path(configured)
        if configured_path.exists():
            return str(configured_path)
        common_paths = [
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
        ]
        for path in common_paths:
            if path.exists():
                return str(path)
        return configured

    def _iter_clip_files(self, folder: Path) -> list[Path]:
        extensions = {extension.lower() for extension in self.config.obs_clip_extensions}
        return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions)

    def _file_is_stable(self, path: Path) -> bool:
        try:
            first = (path.stat().st_size, path.stat().st_mtime_ns)
            time.sleep(self.config.file_stable_seconds)
            second = (path.stat().st_size, path.stat().st_mtime_ns)
        except FileNotFoundError:
            return False
        return first == second and first[0] > 0

    def _file_is_settled(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        return stat.st_size > 0 and (time.time() - stat.st_mtime) >= self.config.file_stable_seconds


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()
    raw = load_json_config(path)
    openai_raw = raw.get("openai", {})
    lmstudio_raw = raw.get("lmstudio", {})
    whisperlive_raw = raw.get("whisperlive", {})
    local_whisper_raw = raw.get("local_whisper", {})
    voice_raw = raw.get("voice", {})
    obs_raw = raw.get("obs", {})
    voice_command_provider = str(raw.get("voice_command_provider", AppConfig.voice_command_provider)).lower()
    if voice_command_provider == "lmstudio":
        print("LM Studio live voice commands are no longer supported; falling back to Vosk.")
        voice_command_provider = "vosk"
    rename_transcription_provider = str(
        raw.get("rename_transcription_provider", AppConfig.rename_transcription_provider)
    ).lower()
    if rename_transcription_provider == "lmstudio":
        print("LM Studio audio transcription is no longer supported; using OpenAI for rename transcription.")
        rename_transcription_provider = "openai"
    legacy_openai_transcription_model = str(
        openai_raw.get("transcription_model", OpenAIConfig.rename_transcription_model)
    )
    audio_devices = normalize_audio_device_values(raw.get("audio_devices", AppConfig.audio_devices))
    audio_device = raw.get("audio_device", AppConfig.audio_device)
    if not audio_devices:
        audio_devices = normalize_audio_device_values(audio_device)
        audio_device = audio_devices[0] if len(audio_devices) == 1 else audio_device
    return AppConfig(
        clip_seconds=int(raw.get("clip_seconds", AppConfig.clip_seconds)),
        segment_seconds=int(raw.get("segment_seconds", AppConfig.segment_seconds)),
        fps=int(raw.get("fps", AppConfig.fps)),
        video_source=str(raw.get("video_source", AppConfig.video_source)),
        monitor_index=int(raw.get("monitor_index", AppConfig.monitor_index)),
        camera_index=int(raw.get("camera_index", AppConfig.camera_index)),
        camera_width=raw.get("camera_width", AppConfig.camera_width),
        camera_height=raw.get("camera_height", AppConfig.camera_height),
        video_device_name=raw.get("video_device_name", AppConfig.video_device_name),
        audio_device_name=raw.get("audio_device_name", AppConfig.audio_device_name),
        audio_device=audio_device,
        audio_devices=audio_devices,
        audio_sample_rate=int(raw.get("audio_sample_rate", AppConfig.audio_sample_rate)),
        audio_channels=int(raw.get("audio_channels", AppConfig.audio_channels)),
        command_check_seconds=int(raw.get("command_check_seconds", AppConfig.command_check_seconds)),
        voice_commands=tuple(raw.get("voice_commands", AppConfig.voice_commands)),
        output_dir=Path(raw.get("output_dir", AppConfig.output_dir)),
        buffer_dir=Path(raw.get("buffer_dir", AppConfig.buffer_dir)),
        obs_output_dir=Path(raw["obs_output_dir"]) if raw.get("obs_output_dir") else None,
        obs_clip_extensions=tuple(raw.get("obs_clip_extensions", AppConfig.obs_clip_extensions)),
        file_stable_seconds=float(raw.get("file_stable_seconds", AppConfig.file_stable_seconds)),
        poll_seconds=float(raw.get("poll_seconds", AppConfig.poll_seconds)),
        ffmpeg_path=str(raw.get("ffmpeg_path", AppConfig.ffmpeg_path)),
        ai_provider=str(raw.get("ai_provider", AppConfig.ai_provider)),
        voice_command_provider=voice_command_provider,
        rename_transcription_provider=rename_transcription_provider,
        rename_transcription_audio_fraction=float(
            raw.get("rename_transcription_audio_fraction", AppConfig.rename_transcription_audio_fraction)
        ),
        name_live_clips=bool(raw.get("name_live_clips", AppConfig.name_live_clips)),
        filename_prefix=str(raw.get("filename_prefix", AppConfig.filename_prefix)),
        filename_suffix=str(raw.get("filename_suffix", AppConfig.filename_suffix)),
        openai=OpenAIConfig(
            api_key=openai_raw.get("api_key", OpenAIConfig.api_key),
            api_key_env=str(openai_raw.get("api_key_env", OpenAIConfig.api_key_env)),
            voice_command_transcription_model=str(
                openai_raw.get("voice_command_transcription_model", legacy_openai_transcription_model)
            ),
            rename_transcription_model=str(
                openai_raw.get("rename_transcription_model", legacy_openai_transcription_model)
            ),
            naming_model=str(openai_raw.get("naming_model", OpenAIConfig.naming_model)),
            max_frames_for_naming=int(openai_raw.get("max_frames_for_naming", OpenAIConfig.max_frames_for_naming)),
        ),
        lmstudio=LMStudioConfig(
            base_url=str(lmstudio_raw.get("base_url", LMStudioConfig.base_url)),
            api_key=lmstudio_raw.get("api_key", LMStudioConfig.api_key),
            api_key_env=str(lmstudio_raw.get("api_key_env", LMStudioConfig.api_key_env)),
            vision_model=str(lmstudio_raw.get("vision_model", LMStudioConfig.vision_model)),
        ),
        whisperlive=WhisperLiveConfig(
            base_url=str(whisperlive_raw.get("base_url", WhisperLiveConfig.base_url)),
            api_key=whisperlive_raw.get("api_key", WhisperLiveConfig.api_key),
            api_key_env=str(whisperlive_raw.get("api_key_env", WhisperLiveConfig.api_key_env)),
            model=str(whisperlive_raw.get("model", WhisperLiveConfig.model)),
            language=str(whisperlive_raw.get("language", WhisperLiveConfig.language)),
        ),
        local_whisper=LocalWhisperConfig(
            model_size=str(local_whisper_raw.get("model_size", LocalWhisperConfig.model_size)),
            device=str(local_whisper_raw.get("device", LocalWhisperConfig.device)),
            compute_type=str(local_whisper_raw.get("compute_type", LocalWhisperConfig.compute_type)),
            cpu_threads=int(local_whisper_raw.get("cpu_threads", LocalWhisperConfig.cpu_threads)),
        ),
        voice=VoiceConfig(
            provider=str(voice_raw.get("provider", VoiceConfig.provider)),
            vosk_model_path=resolve_vosk_model_path(
                bundled_path(Path(voice_raw.get("vosk_model_path", VoiceConfig.vosk_model_path)))
            ),
            rename_vosk_model_path=resolve_vosk_model_path(
                bundled_path(Path(voice_raw.get("rename_vosk_model_path", VoiceConfig.rename_vosk_model_path)))
            ),
            trigger_cooldown_seconds=float(
                voice_raw.get("trigger_cooldown_seconds", VoiceConfig.trigger_cooldown_seconds)
            ),
            clip_action=str(voice_raw.get("clip_action", VoiceConfig.clip_action)),
            enable_obs_scene_source_switching=bool(
                voice_raw.get(
                    "enable_obs_scene_source_switching",
                    VoiceConfig.enable_obs_scene_source_switching,
                )
            ),
            require_wake_phrase=bool(voice_raw.get("require_wake_phrase", VoiceConfig.require_wake_phrase)),
            wake_phrases=tuple(voice_raw.get("wake_phrases", VoiceConfig.wake_phrases)),
            wake_listen_seconds=float(voice_raw.get("wake_listen_seconds", VoiceConfig.wake_listen_seconds)),
        ),
        obs=OBSConfig(
            host=str(obs_raw.get("host", OBSConfig.host)),
            port=int(obs_raw.get("port", OBSConfig.port)),
            password=obs_raw.get("password", OBSConfig.password),
            password_env=str(obs_raw.get("password_env", OBSConfig.password_env)),
            request_timeout_seconds=float(
                obs_raw.get("request_timeout_seconds", OBSConfig.request_timeout_seconds)
            ),
        ),
    )


def sample_video_frames(video_path: Path, max_frames: int) -> list[str]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        return []
    indices = np.linspace(0, max(0, frame_count - 1), num=min(max_frames, frame_count), dtype=int)
    frames: list[str] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        scale = min(1.0, 960 / max(width, height))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            frames.append(f"data:image/jpeg;base64,{b64}")
    capture.release()
    return frames


def sanitize_filename(value: str, fallback: str = "clip") -> str:
    value = re.sub(r"[^\w\s.-]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value.strip())
    value = value.strip("._-")
    return value[:80] or fallback


def extract_chat_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
            else:
                value = getattr(item, "text", None) or getattr(item, "content", None)
                if value:
                    parts.append(str(value))
        return " ".join(parts)
    return str(content or "")


def clean_generated_title(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    value = re.sub(r"(?is)<think>.*?</think>", "", value).strip()
    value = re.sub(r"(?i)^title\s*:\s*", "", value).strip()
    value = value.splitlines()[0].strip() if value else ""
    value = re.sub(r"\.(mp4|mkv|mov|flv)$", "", value, flags=re.IGNORECASE).strip()
    return sanitize_filename(value, fallback="") if value else ""


def extract_title_from_reasoning(message: Any) -> str:
    reasoning = str(getattr(message, "reasoning_content", "") or "").strip()
    if not reasoning:
        return ""

    base = ""
    base_match = re.search(r"(?im)^\s*(?:\*\s*)?Base title:\s*(.+)$", reasoning)
    if base_match:
        base = base_match.group(1).strip()
    else:
        game_match = re.search(r"(?im)^\s*\*\s*Game:\s*(.+)$", reasoning)
        if game_match:
            base = f"{game_match.group(1).strip()} Clip"

    if not base:
        return ""

    specs: list[str] = []
    for pattern in (
        r"\b(?:RTX|GTX|RX|Radeon|Arc)\s*[A-Za-z0-9 -]{2,12}\b",
        r"\b(?:R\d|Ryzen|i[3579]|Core)\s*[A-Za-z0-9 -]{2,12}\b",
        r"\b\d{3,4}x\d{3,4}\b",
        r"\b(?:720p|1080p|1440p|2160p|4K)\b",
        r"\b(?:Vulkan|DX11|DX12|OpenGL)\b",
    ):
        for match in re.findall(pattern, reasoning, flags=re.IGNORECASE):
            value = re.sub(r"\s+", " ", str(match).strip())
            if value and value.lower() not in {item.lower() for item in specs}:
                specs.append(value)

    title = " ".join([base, *specs[:4]])
    recovered = sanitize_filename(title, fallback="")
    if recovered:
        print(f"Recovered LM Studio title from reasoning: {recovered}", flush=True)
    return recovered


def normalize_obs_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique path for {path}")


def list_video_devices(max_index: int = 10, snapshot_dir: Path | None = None) -> None:
    found = False
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            continue
        ok, frame = capture.read()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        capture.release()
        if ok and frame is not None:
            frame_height, frame_width = frame.shape[:2]
            width = width or frame_width
            height = height or frame_height
            snapshot_note = ""
            if snapshot_dir:
                snapshot_path = snapshot_dir / f"video_device_{index}.jpg"
                cv2.imwrite(str(snapshot_path), frame)
                snapshot_note = f" -> {snapshot_path}"
            print(f"{index}: opened ({width}x{height}, reported {fps:.1f} fps){snapshot_note}")
            found = True
    if not found:
        print("No camera-style video devices opened. Start OBS Virtual Camera and try again.")


def list_named_capture_devices(ffmpeg_path: str) -> None:
    print("Windows capture-device candidates:")
    pnp_command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-PnpDevice -PresentOnly | "
            "Where-Object { $_.Class -in @('Camera','Image','MEDIA') -or "
            "$_.FriendlyName -match 'HD60|Signal|Capture|Video|Camera|Cam|OBS|USB' } | "
            "Sort-Object Class,FriendlyName | "
            "Format-Table -AutoSize Class,FriendlyName,Status"
        ),
    ]
    try:
        result = subprocess.run(pnp_command, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip() or result.stderr.strip()
        print(output if output else "No Windows PnP capture devices were returned.")
    except Exception as exc:
        print(f"Windows PnP listing failed: {exc}")

    print("\nFFmpeg DirectShow names:")
    command = [ffmpeg_path, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        output = (result.stderr or result.stdout).strip()
        print(output if output else "FFmpeg returned no DirectShow device list.")
    except Exception as exc:
        print(f"FFmpeg DirectShow listing failed: {exc}")


def update_config_file(path: Path, updates: dict[str, Any]) -> None:
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    for key, value in updates.items():
        data[key] = value
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {path}")


def run_whisperlive_server(
    *,
    host: str = "0.0.0.0",
    port: int = 9090,
    rest_port: int = 8000,
    backend: str = "faster_whisper",
    model: str = "base.en",
    max_clients: int = 2,
    max_connection_time: int = 43200,
    cache_path: str = "models/whisper-live-cache",
    api_key: str | None = None,
) -> None:
    try:
        from whisper_live.server import TranscriptionServer
    except Exception as exc:
        print(f"WhisperLive is not available in this environment: {exc}", flush=True)
        print("Run .\\run_whisperlive_server.bat from a source checkout, or rebuild the EXE with whisper-live installed.", flush=True)
        raise SystemExit(1) from exc

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print("Starting WhisperLive server", flush=True)
    print(f"WebSocket: ws://localhost:{port}", flush=True)
    print(f"REST: http://localhost:{rest_port}/v1", flush=True)
    print(f"Backend: {backend}", flush=True)
    print(f"Model: {model}", flush=True)
    server = TranscriptionServer()
    server.run(
        host,
        port=port,
        backend=backend,
        faster_whisper_custom_model_path=model,
        single_model=True,
        max_clients=max_clients,
        max_connection_time=max_connection_time,
        cache_path=cache_path,
        rest_port=rest_port,
        enable_rest=True,
        api_key=api_key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling screen/audio clipper with voice-triggered naming.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--live-clipper", action="store_true", help="Start the live voice-triggered clipper.")
    parser.add_argument("--whisperlive-server", action="store_true", help="Run the local WhisperLive server.")
    parser.add_argument("--whisperlive-host", default="0.0.0.0", help="Host for --whisperlive-server.")
    parser.add_argument("--whisperlive-port", type=int, default=9090, help="WebSocket port for --whisperlive-server.")
    parser.add_argument("--whisperlive-rest-port", type=int, default=8000, help="REST port for --whisperlive-server.")
    parser.add_argument("--whisperlive-backend", default="faster_whisper", help="Backend for --whisperlive-server.")
    parser.add_argument("--whisperlive-model", help="WhisperLive/faster-whisper model for --whisperlive-server.")
    parser.add_argument("--whisperlive-max-clients", type=int, default=2, help="Max clients for --whisperlive-server.")
    parser.add_argument(
        "--whisperlive-max-connection-time",
        type=int,
        default=43200,
        help="Max client lifetime in seconds for --whisperlive-server.",
    )
    parser.add_argument(
        "--whisperlive-cache-path",
        default="models/whisper-live-cache",
        help="Model cache path for --whisperlive-server.",
    )
    parser.add_argument("--whisperlive-api-key", help="Optional API key for --whisperlive-server.")
    parser.add_argument("--list-audio-devices", action="store_true", help="Print available audio devices and exit.")
    parser.add_argument("--list-video-devices", action="store_true", help="Probe camera-style video devices and exit.")
    parser.add_argument("--list-capture-devices", action="store_true", help="List named Windows/DirectShow capture devices.")
    parser.add_argument("--max-video-index", type=int, default=10, help="Highest video device index to probe.")
    parser.add_argument(
        "--snapshot-video-devices",
        action="store_true",
        help="Save one JPEG from each opened video device while probing.",
    )
    parser.add_argument("--use-hd60", action="store_true", help="Configure DirectShow capture for NZXT Signal HD60.")
    parser.add_argument("--set-directshow-video", help="Set the named DirectShow video device to watch.")
    parser.add_argument("--set-directshow-audio", help="Set the named DirectShow audio device for captured clips.")
    parser.add_argument("--set-mic-device", help="Set the mic/input device used for voice commands and local clip audio.")
    parser.add_argument(
        "--set-audio-devices",
        nargs="+",
        help="Set multiple input devices used for voice commands and local clip audio.",
    )
    parser.add_argument("--set-obs-output-dir", help="Set the OBS recording/replay folder to watch.")
    parser.add_argument("--watch-obs-clips", action="store_true", help="Watch OBS output folder and rename new clips.")
    parser.add_argument("--rename-existing", action="store_true", help="When watching, also rename existing clips.")
    parser.add_argument(
        "--batch-rename-obs-clips",
        action="store_true",
        help="Rename stable clips already present in the configured OBS output folder and exit.",
    )
    parser.add_argument(
        "--rename-folder",
        help="Folder to batch rename with --batch-rename-obs-clips instead of the configured OBS output folder.",
    )
    parser.add_argument("--rename-file", help="Rename one existing video file using frame/audio context.")
    parser.add_argument("--save-obs-replay-buffer", action="store_true", help="Tell OBS to save the replay buffer.")
    parser.add_argument("--start-obs-replay-buffer", action="store_true", help="Tell OBS to start the replay buffer.")
    parser.add_argument("--stop-obs-replay-buffer", action="store_true", help="Tell OBS to stop the replay buffer.")
    parser.add_argument("--start-obs-recording", action="store_true", help="Tell OBS to start recording.")
    parser.add_argument("--stop-obs-recording", action="store_true", help="Tell OBS to stop recording.")
    parser.add_argument("--list-obs-scenes-sources", action="store_true", help="List OBS scenes and scene-item sources.")
    parser.add_argument("--switch-obs-to", help="Switch OBS to a scene or source by name.")
    parser.add_argument(
        "--voice-command-smoke-test",
        metavar="TEXT",
        help="Match a voice-command phrase and exit without using the microphone, OBS, or capture devices.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    if args.whisperlive_server:
        run_whisperlive_server(
            host=args.whisperlive_host,
            port=args.whisperlive_port,
            rest_port=args.whisperlive_rest_port,
            backend=args.whisperlive_backend,
            model=args.whisperlive_model or config.whisperlive.model,
            max_clients=args.whisperlive_max_clients,
            max_connection_time=args.whisperlive_max_connection_time,
            cache_path=args.whisperlive_cache_path,
            api_key=args.whisperlive_api_key or config.whisperlive.api_key,
        )
        return

    if args.voice_command_smoke_test is not None:
        clipper = RollingClipper(config)
        action = clipper._voice_command_action_after_wake(args.voice_command_smoke_test)
        if action is None:
            print(f"No voice command matched: {args.voice_command_smoke_test}")
            raise SystemExit(1)
        if action.target:
            print(f"Voice command matched: {action.kind} -> {action.target}")
        else:
            print(f"Voice command matched: {action.kind}")
        return

    if args.list_audio_devices:
        print(sd.query_devices())
        return
    if args.list_video_devices:
        snapshot_dir = Path(".capture_buffer") / "video_device_probe" if args.snapshot_video_devices else None
        list_video_devices(args.max_video_index, snapshot_dir=snapshot_dir)
        return
    if args.list_capture_devices:
        ffmpeg_path = RollingClipper(config)._resolve_ffmpeg(config.ffmpeg_path)
        list_named_capture_devices(ffmpeg_path)
        return

    if args.use_hd60:
        update_config_file(
            config_path,
            {
                "video_source": "directshow",
                "video_device_name": "NZXT Signal HD60 Video",
                "audio_device_name": None,
                "audio_device": None,
                "camera_width": 1920,
                "camera_height": 1080,
            },
        )
        return

    config_updates: dict[str, Any] = {}
    if args.set_directshow_video:
        config_updates["video_source"] = "directshow"
        config_updates["video_device_name"] = args.set_directshow_video
    if args.set_directshow_audio:
        config_updates["audio_device_name"] = args.set_directshow_audio
    if args.set_mic_device:
        try:
            config_updates["audio_device"] = int(args.set_mic_device)
        except ValueError:
            config_updates["audio_device"] = args.set_mic_device
        config_updates["audio_devices"] = []
    if args.set_audio_devices:
        parsed_devices: list[int | str] = []
        for device in args.set_audio_devices:
            try:
                parsed_devices.append(int(device))
            except ValueError:
                parsed_devices.append(device)
        config_updates["audio_devices"] = parsed_devices
        config_updates["audio_device"] = None
    if args.set_obs_output_dir:
        config_updates["obs_output_dir"] = args.set_obs_output_dir
    if config_updates:
        update_config_file(config_path, config_updates)
        return

    clipper = RollingClipper(config)
    if args.rename_file:
        renamed = clipper.rename_clip_file(Path(args.rename_file))
        print(f"Result: {renamed}")
        return
    if args.save_obs_replay_buffer:
        clipper.save_obs_replay_buffer()
        return
    if args.start_obs_replay_buffer:
        clipper.start_obs_replay_buffer()
        return
    if args.stop_obs_replay_buffer:
        clipper.stop_obs_replay_buffer()
        return
    if args.start_obs_recording:
        clipper.start_obs_recording()
        return
    if args.stop_obs_recording:
        clipper.stop_obs_recording()
        return
    if args.list_obs_scenes_sources:
        clipper.list_obs_scenes_sources()
        return
    if args.switch_obs_to:
        clipper.switch_obs_scene_or_source(args.switch_obs_to)
        return
    if args.watch_obs_clips:
        clipper.watch_obs_folder(rename_existing=args.rename_existing)
        return
    if args.batch_rename_obs_clips:
        clipper.batch_rename_obs_clips(Path(args.rename_folder) if args.rename_folder else None)
        return
    if args.live_clipper:
        print("Live clipper worker starting.", flush=True)
        clipper.run()
        return
    clipper.run()


if __name__ == "__main__":
    main()
