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


BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def bundled_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    bundled = BUNDLE_DIR / path
    return bundled if bundled.exists() else path


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


@dataclass
class OpenAIConfig:
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    transcription_model: str = "gpt-4o-mini-transcribe"
    naming_model: str = "gpt-4.1-mini"
    max_frames_for_naming: int = 8


@dataclass
class LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    api_key: str | None = None
    api_key_env: str = "LMSTUDIO_API_KEY"
    vision_model: str = "qwen2.5-vl-7b-instruct"


@dataclass
class VoiceConfig:
    provider: str = "vosk"
    vosk_model_path: Path = Path("models/vosk-model-small-en-us-0.15")
    trigger_cooldown_seconds: float = 2.0
    clip_action: str = "obs_replay_buffer"
    enable_obs_scene_source_switching: bool = False


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
    filename_prefix: str = ""
    filename_suffix: str = ""
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
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


class RollingClipper:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.segments: deque[Segment] = deque()
        self.segment_lock = threading.Lock()
        self.audio_chunks: queue.Queue[Any] = queue.Queue()
        self.stop_event = threading.Event()
        self.openai_client = self._build_openai_client()
        self.lmstudio_client = self._build_lmstudio_client()
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
        token = self.config.lmstudio.api_key or os.getenv(self.config.lmstudio.api_key_env)
        if not token:
            return None
        return OpenAI(base_url=self.config.lmstudio.base_url, api_key=token)

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
        if self.config.voice_command_provider.lower() != "vosk" and self.openai_client is None:
            print("Voice command transcription is disabled because OPENAI_API_KEY is not set. Use the UI Clip Now button.")
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

        while not self.stop_event.is_set():
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
        self._openai_command_loop()

    def _openai_command_loop(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(self.config.command_check_seconds)
            if self.openai_client is None:
                continue
            chunks = self._drain_recent_audio()
            if not chunks:
                continue
            text = self._transcribe_audio_array(chunks, prefix="command")
            if not text:
                continue
            normalized = text.lower()
            action = self._voice_command_action(normalized)
            if action:
                print(f"Voice command heard: {text.strip()}")
                try:
                    self.handle_voice_action(action)
                except Exception as exc:
                    print(f"Could not run voice command: {exc}")

    def _vosk_command_loop(self) -> None:
        if Model is None or KaldiRecognizer is None:
            print("Vosk is not installed. Run pip install -r requirements.txt.")
            return
        model_path = self.config.voice.vosk_model_path
        if not model_path.exists():
            print(f"Vosk model not found: {model_path}")
            return

        print(f"Local voice commands enabled with Vosk model: {model_path}")
        model = Model(str(model_path))
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
            action = self._voice_command_action(normalized)
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

    def _is_voice_command(self, text: str) -> bool:
        return self._voice_command_action(text) is not None

    def _voice_command_action(self, text: str) -> VoiceAction | None:
        normalized = text.lower().strip()
        if any(command in normalized for command in START_REPLAY_BUFFER_COMMANDS):
            return VoiceAction("start_replay_buffer")
        if any(command in normalized for command in STOP_REPLAY_BUFFER_COMMANDS):
            return VoiceAction("stop_replay_buffer")
        if any(command in normalized for command in START_RECORDING_COMMANDS):
            return VoiceAction("start_recording")
        if any(command in normalized for command in STOP_RECORDING_COMMANDS):
            return VoiceAction("stop_recording")
        if self.config.voice.enable_obs_scene_source_switching:
            target = self._extract_obs_switch_target(normalized)
            if target:
                return VoiceAction("switch_obs_scene_or_source", target)
        if any(command in normalized for command in self.config.voice_commands):
            return VoiceAction("clip")
        words = set(re.findall(r"[a-z]+", normalized))
        if words.intersection({"clip", "save", "capture"}) or {"record", "that"}.issubset(words):
            return VoiceAction("clip")
        return None

    def _extract_obs_switch_target(self, normalized: str) -> str | None:
        for pattern in OBS_SWITCH_PATTERNS:
            match = pattern.search(normalized)
            if not match:
                continue
            target = re.sub(r"\b(?:please|the|scene|source)\b", " ", match.group("target"))
            target = re.sub(r"\s+", " ", target).strip(" .,!?:;")
            if target:
                return target
        return None

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
        status = client.get_replay_buffer_status()
        if getattr(status, "output_active", False):
            print("OBS replay buffer is already running.")
            return
        client.start_replay_buffer()
        print("OBS replay buffer start requested.")

    def stop_obs_replay_buffer(self) -> None:
        client = self._obs_client()
        status = client.get_replay_buffer_status()
        if not getattr(status, "output_active", False):
            print("OBS replay buffer is already stopped.")
            return
        client.stop_replay_buffer()
        print("OBS replay buffer stop requested.")

    def start_obs_recording(self) -> None:
        client = self._obs_client()
        status = client.get_record_status()
        if getattr(status, "output_active", False):
            print("OBS recording is already running.")
            return
        client.start_record()
        print("OBS recording start requested.")

    def stop_obs_recording(self) -> None:
        client = self._obs_client()
        status = client.get_record_status()
        if not getattr(status, "output_active", False):
            print("OBS recording is already stopped.")
            return
        client.stop_record()
        print("OBS recording stop requested.")

    def switch_obs_scene_or_source(self, target: str) -> None:
        client = self._obs_client()
        inventory = self._read_obs_scene_source_inventory(client)
        scene = self._best_obs_name_match(target, inventory["scenes"], "scene")
        if scene:
            client.set_current_program_scene(scene_name=scene)
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
        client.set_current_program_scene(scene_name=scene_name)
        if selected.get("scene_item_id") is not None and selected.get("scene_item_enabled") is False:
            client.set_scene_item_enabled(
                scene_name=scene_name,
                scene_item_id=selected["scene_item_id"],
                scene_item_enabled=True,
            )
        print(f"OBS source selected: {source} in scene {scene_name}")

    def _read_obs_scene_source_inventory(self, client: Any) -> dict[str, Any]:
        scene_result = client.get_scene_list()
        scenes_raw = self._obs_value(scene_result, "scenes", default=[])
        current_scene = self._obs_value(scene_result, "current_program_scene_name", "currentProgramSceneName")
        scenes = [self._obs_value(scene, "scene_name", "sceneName") for scene in scenes_raw]
        scenes = [scene for scene in scenes if scene]

        inputs_result = client.get_input_list()
        inputs_raw = self._obs_value(inputs_result, "inputs", default=[])
        inputs = [self._obs_value(input_item, "input_name", "inputName") for input_item in inputs_raw]
        inputs = [input_name for input_name in inputs if input_name]

        scene_items: list[dict[str, Any]] = []
        for scene_name in scenes:
            try:
                items_result = client.get_scene_item_list(scene_name=scene_name)
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
        status = client.get_replay_buffer_status()
        if not getattr(status, "output_active", False):
            raise RuntimeError("OBS replay buffer is not active. Start Replay Buffer in OBS first.")
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

    def batch_rename_obs_clips(self) -> None:
        if not self.config.obs_output_dir:
            raise ValueError("obs_output_dir is required for --batch-rename-obs-clips")
        folder = self.config.obs_output_dir
        folder.mkdir(parents=True, exist_ok=True)
        paths = self._iter_clip_files(folder)
        if not paths:
            print(f"No OBS clips found in {folder}.", flush=True)
            return
        print(f"Batch scanning {len(paths)} OBS clip(s) in {folder}.", flush=True)
        renamed_count = 0
        skipped_count = 0
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
                print(f"[{index}/{len(paths)}] Could not rename {path.name}: {exc}", flush=True)
        print(
            f"Batch rename complete. Renamed {renamed_count}, skipped {skipped_count}, scanned {len(paths)} clip(s).",
            flush=True,
        )

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
            sanitize_filename(self.config.filename_prefix),
            sanitize_filename(generated_name),
            sanitize_filename(self.config.filename_suffix),
        ]
        return "_".join(part for part in parts if part) or "clip"

    def _suggest_name(self, video_path: Path, audio_path: Path | None) -> str | None:
        transcript = self._transcribe_file(audio_path) if audio_path else ""
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

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Name this video clip. Return only a concise filesystem-safe title, "
                    "no extension. Use visible video context and transcript. If an RTSS, MSI "
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
                model=self.config.lmstudio.vision_model,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=32,
            )
            message = response.choices[0].message.content or ""
            return message.strip()
        except Exception as exc:
            print(f"LM Studio naming failed: {exc}")
            return None

    def _transcribe_file(self, audio_path: Path) -> str:
        if self.openai_client is None or not audio_path.exists() or audio_path.stat().st_size == 0:
            return ""
        try:
            with audio_path.open("rb") as audio_file:
                result = self.openai_client.audio.transcriptions.create(
                    model=self.config.openai.transcription_model,
                    file=audio_file,
                )
            return getattr(result, "text", "") or ""
        except Exception as exc:
            print(f"OpenAI transcription failed: {exc}")
            return ""

    def _transcribe_audio_array(self, chunks: list[np.ndarray], prefix: str) -> str:
        if self.openai_client is None:
            return ""
        with tempfile.NamedTemporaryFile(prefix=f"{prefix}_", suffix=".wav", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            self._write_wav(temp_path, chunks)
            return self._transcribe_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _drain_recent_audio(self) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        while True:
            try:
                item = self.audio_chunks.get_nowait()
                chunks.append(item[1] if isinstance(item, tuple) and len(item) == 2 else item)
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
            str(audio_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return audio_path

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

    def _start_audio_recorders(self) -> list[AudioRecorder]:
        recorders: list[AudioRecorder] = []
        for device in self._configured_audio_devices():
            frames: list[np.ndarray] = []

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
                chunk = indata.copy()
                captured_frames.append(chunk)
                self.audio_chunks.put((captured_device, chunk))

            try:
                stream = sd.InputStream(
                    samplerate=self.config.audio_sample_rate,
                    channels=self.config.audio_channels,
                    device=device,
                    dtype="int16",
                    callback=audio_callback,
                )
                stream.start()
                recorders.append(AudioRecorder(device=device, frames=frames, stream=stream))
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
    raw = json.loads(path.read_text(encoding="utf-8"))
    openai_raw = raw.get("openai", {})
    lmstudio_raw = raw.get("lmstudio", {})
    voice_raw = raw.get("voice", {})
    obs_raw = raw.get("obs", {})
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
        audio_device=raw.get("audio_device", AppConfig.audio_device),
        audio_devices=tuple(raw.get("audio_devices", AppConfig.audio_devices)),
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
        voice_command_provider=str(raw.get("voice_command_provider", AppConfig.voice_command_provider)),
        filename_prefix=str(raw.get("filename_prefix", AppConfig.filename_prefix)),
        filename_suffix=str(raw.get("filename_suffix", AppConfig.filename_suffix)),
        openai=OpenAIConfig(
            api_key=openai_raw.get("api_key", OpenAIConfig.api_key),
            api_key_env=str(openai_raw.get("api_key_env", OpenAIConfig.api_key_env)),
            transcription_model=str(openai_raw.get("transcription_model", OpenAIConfig.transcription_model)),
            naming_model=str(openai_raw.get("naming_model", OpenAIConfig.naming_model)),
            max_frames_for_naming=int(openai_raw.get("max_frames_for_naming", OpenAIConfig.max_frames_for_naming)),
        ),
        lmstudio=LMStudioConfig(
            base_url=str(lmstudio_raw.get("base_url", LMStudioConfig.base_url)),
            api_key=lmstudio_raw.get("api_key", LMStudioConfig.api_key),
            api_key_env=str(lmstudio_raw.get("api_key_env", LMStudioConfig.api_key_env)),
            vision_model=str(lmstudio_raw.get("vision_model", LMStudioConfig.vision_model)),
        ),
        voice=VoiceConfig(
            provider=str(voice_raw.get("provider", VoiceConfig.provider)),
            vosk_model_path=bundled_path(Path(voice_raw.get("vosk_model_path", VoiceConfig.vosk_model_path))),
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


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^\w\s.-]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value.strip())
    value = value.strip("._-")
    return value[:80] or "clip"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling screen/audio clipper with voice-triggered naming.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
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
    parser.add_argument("--rename-file", help="Rename one existing video file using frame/audio context.")
    parser.add_argument("--save-obs-replay-buffer", action="store_true", help="Tell OBS to save the replay buffer.")
    parser.add_argument("--start-obs-replay-buffer", action="store_true", help="Tell OBS to start the replay buffer.")
    parser.add_argument("--stop-obs-replay-buffer", action="store_true", help="Tell OBS to stop the replay buffer.")
    parser.add_argument("--start-obs-recording", action="store_true", help="Tell OBS to start recording.")
    parser.add_argument("--stop-obs-recording", action="store_true", help="Tell OBS to stop recording.")
    parser.add_argument("--list-obs-scenes-sources", action="store_true", help="List OBS scenes and scene-item sources.")
    parser.add_argument("--switch-obs-to", help="Switch OBS to a scene or source by name.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

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
        clipper.batch_rename_obs_clips()
        return
    clipper.run()


if __name__ == "__main__":
    main()
