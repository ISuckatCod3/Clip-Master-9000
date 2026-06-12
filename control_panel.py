from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
import ctypes
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import mss
import sounddevice as sd


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
CONFIG_PATH = APP_DIR / "config.json"
SCRIPT_PATH = APP_DIR / "live_video_interpreter.py"
ICON_PATH = BUNDLE_DIR / "assets" / "app.ico"
APP_NAME = "Clip Master 9000"
REPO_URL = "https://github.com/ISuckatCod3/Clip-Master-9000"
WORKER_ARG_SETS = {
    "Start Live Clipper": ["--live-clipper"],
    "Save OBS Replay": ["--save-obs-replay-buffer"],
    "Start OBS Renamer": ["--watch-obs-clips"],
    "Batch Rename Existing": ["--batch-rename-obs-clips"],
    "Rename One Clip": ["--rename-file", "<selected-file>"],
}

DARK_COLORS = {
    "background": "#111318",
    "panel": "#181b22",
    "panel_alt": "#20242d",
    "field": "#0d0f14",
    "border": "#343946",
    "text": "#ffa844",
    "muted": "#a7adbb",
    "accent": "#2f8cff",
    "accent_active": "#1fa8d1",
    "danger": "#d64f4f",
    "selection": "#353972",
}


DEFAULT_CONFIG = {
    "clip_seconds": 45,
    "segment_seconds": 5,
    "fps": 12,
    "video_source": "screen",
    "monitor_index": 1,
    "camera_index": 0,
    "camera_width": None,
    "camera_height": None,
    "video_device_name": None,
    "audio_device_name": None,
    "audio_device": None,
    "audio_devices": [],
    "audio_sample_rate": 16000,
    "audio_channels": 1,
    "command_check_seconds": 4,
    "voice_commands": [
        "clip that",
        "save that",
        "record that",
        "capture that",
        "start replay buffer",
        "stop replay buffer",
        "start recording",
        "stop recording",
    ],
    "output_dir": "clips",
    "buffer_dir": ".capture_buffer",
    "obs_output_dir": None,
    "obs_clip_extensions": [".mp4", ".mkv", ".mov", ".flv"],
    "file_stable_seconds": 3,
    "poll_seconds": 2,
    "ffmpeg_path": "ffmpeg",
    "ai_provider": "openai",
    "voice_command_provider": "vosk",
    "rename_transcription_provider": "local_whisper",
    "name_live_clips": False,
    "filename_prefix": "",
    "filename_suffix": "",
    "openai": {
        "api_key": None,
        "api_key_env": "OPENAI_API_KEY",
        "voice_command_transcription_model": "gpt-4o-mini-transcribe",
        "rename_transcription_model": "gpt-4o-mini-transcribe",
        "naming_model": "gpt-4.1-mini",
        "max_frames_for_naming": 8,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key_env": "LMSTUDIO_API_KEY",
        "vision_model": "qwen3-vl-2b",
    },
    "local_whisper": {
        "model_size": "base.en",
        "device": "auto",
        "compute_type": "int8",
        "cpu_threads": 0,
    },
    "voice": {
        "provider": "vosk",
        "vosk_model_path": "models/vosk-model-en-us-0.22-lgraph",
        "rename_vosk_model_path": "models/vosk-model-en-us-0.22-lgraph",
        "trigger_cooldown_seconds": 2,
        "clip_action": "obs_replay_buffer",
        "enable_obs_scene_source_switching": False,
    },
    "obs": {
        "host": "localhost",
        "port": 4455,
        "password_env": "OBS_WEBSOCKET_PASSWORD",
        "request_timeout_seconds": 5,
    },
}


class ControlPanel(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("980x720")
        self.minsize(860, 620)
        self.configure(background=DARK_COLORS["background"])
        self.apply_window_icon()
        self.enable_windows_dark_title_bar()
        self.configure_dark_style()
        self.process: subprocess.Popen[str] | None = None
        self.video_devices: list[str] = []
        self.audio_devices: list[tuple[int, str]] = []
        self.monitors: list[tuple[int, str]] = []
        self.config = self.load_config()

        self.video_source = tk.StringVar(value=self.config.get("video_source", "screen"))
        self.video_device = tk.StringVar(value=self.config.get("video_device_name") or "")
        self.monitor = tk.StringVar()
        self.ffmpeg_path = tk.StringVar(value=self.config.get("ffmpeg_path", "ffmpeg"))
        self.obs_output_dir = tk.StringVar(value=self.config.get("obs_output_dir") or "")
        self.clip_seconds = tk.StringVar(value=str(self.config.get("clip_seconds", 45)))
        self.fps = tk.StringVar(value=str(self.config.get("fps", 12)))
        self.ai_provider = tk.StringVar(value=self.config.get("ai_provider", "lmstudio"))
        self.rename_transcription_provider = tk.StringVar(
            value=self.config.get("rename_transcription_provider", "local_whisper")
        )
        self.name_live_clips = tk.BooleanVar(value=bool(self.config.get("name_live_clips", False)))
        self.filename_prefix = tk.StringVar(value=self.config.get("filename_prefix", ""))
        self.filename_suffix = tk.StringVar(value=self.config.get("filename_suffix", ""))
        openai = self.config.get("openai", {})
        self.openai_api_key = tk.StringVar(value=openai.get("api_key") or "")
        self.openai_key_env = tk.StringVar(value=openai.get("api_key_env", "OPENAI_API_KEY"))
        legacy_transcription_model = openai.get("transcription_model", "gpt-4o-mini-transcribe")
        self.openai_voice_command_transcription_model = tk.StringVar(
            value=openai.get("voice_command_transcription_model", legacy_transcription_model)
        )
        self.openai_rename_transcription_model = tk.StringVar(
            value=openai.get("rename_transcription_model", legacy_transcription_model)
        )
        self.openai_naming_model = tk.StringVar(value=openai.get("naming_model", "gpt-4.1-mini"))
        self.openai_max_frames = tk.StringVar(value=str(openai.get("max_frames_for_naming", 8)))
        lmstudio = self.config.get("lmstudio", {})
        self.lmstudio_api_key = tk.StringVar(value=lmstudio.get("api_key") or "")
        self.lmstudio_base_url = tk.StringVar(value=lmstudio.get("base_url", "http://localhost:1234/v1"))
        self.lmstudio_model = tk.StringVar(value=lmstudio.get("vision_model", "qwen2.5-vl-7b-instruct"))
        self.lmstudio_key_env = tk.StringVar(value=lmstudio.get("api_key_env", "LMSTUDIO_API_KEY"))
        local_whisper = self.config.get("local_whisper", {})
        self.local_whisper_model = tk.StringVar(value=local_whisper.get("model_size", "base.en"))
        self.local_whisper_device = tk.StringVar(value=local_whisper.get("device", "auto"))
        self.local_whisper_compute_type = tk.StringVar(value=local_whisper.get("compute_type", "int8"))
        self.local_whisper_cpu_threads = tk.StringVar(value=str(local_whisper.get("cpu_threads", 0)))
        voice = self.config.get("voice", {})
        self.voice_provider = tk.StringVar(value=self.config.get("voice_command_provider", voice.get("provider", "vosk")))
        self.vosk_model_path = tk.StringVar(value=voice.get("vosk_model_path", "models/vosk-model-en-us-0.22-lgraph"))
        self.rename_vosk_model_path = tk.StringVar(
            value=voice.get("rename_vosk_model_path", "models/vosk-model-en-us-0.22-lgraph")
        )
        self.clip_action = tk.StringVar(value=voice.get("clip_action", "obs_replay_buffer"))
        self.enable_obs_scene_source_switching = tk.BooleanVar(
            value=bool(voice.get("enable_obs_scene_source_switching", False))
        )
        obs = self.config.get("obs", {})
        self.obs_host = tk.StringVar(value=obs.get("host", "localhost"))
        self.obs_port = tk.StringVar(value=str(obs.get("port", 4455)))
        configured_password_env = obs.get("password_env", "OBS_WEBSOCKET_PASSWORD")
        if configured_password_env != "OBS_WEBSOCKET_PASSWORD" and not os.getenv(configured_password_env, ""):
            self.obs_password = tk.StringVar(value=configured_password_env)
            self.obs_password_env = tk.StringVar(value="OBS_WEBSOCKET_PASSWORD")
        else:
            self.obs_password_env = tk.StringVar(value="OBS_WEBSOCKET_PASSWORD")
            self.obs_password = tk.StringVar(value=os.getenv("OBS_WEBSOCKET_PASSWORD", ""))
        self.status = tk.StringVar(value="Idle")
        self.ai_provider.trace_add("write", lambda *_args: self.update_ai_provider_fields())
        self.voice_provider.trace_add("write", lambda *_args: self.update_ai_provider_fields())
        self.rename_transcription_provider.trace_add("write", lambda *_args: self.update_ai_provider_fields())

        self.build_ui()
        self.update_ai_provider_fields()
        self.refresh_monitors()
        self.refresh_audio_devices()
        self.refresh_video_devices()
        self.load_audio_selection()
        self.fit_window_to_content()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def apply_window_icon(self) -> None:
        if ICON_PATH.exists():
            try:
                self.iconbitmap(default=str(ICON_PATH))
            except tk.TclError:
                pass

    def enable_windows_dark_title_bar(self) -> None:
        if sys.platform != "win32":
            return
        try:
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            value = ctypes.c_int(1)
            for attribute in (20, 19):
                result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attribute,
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                )
                if result == 0:
                    break
        except Exception:
            pass

    def fit_window_to_content(self) -> None:
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        requested_width = max(980, self.winfo_reqwidth() + 24)
        requested_height = max(720, self.winfo_reqheight() + 24)
        max_width = max(860, screen_width - 80)
        max_height = max(620, screen_height - 120)
        width = min(requested_width, max_width)
        height = min(requested_height, max_height)
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def configure_dark_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        colors = DARK_COLORS
        self.option_add("*Font", ("Segoe UI", 10))
        self.option_add("*Listbox.Background", colors["field"])
        self.option_add("*Listbox.Foreground", colors["text"])
        self.option_add("*Listbox.SelectBackground", colors["selection"])
        self.option_add("*Listbox.SelectForeground", colors["text"])
        self.option_add("*Listbox.HighlightColor", colors["accent"])
        self.option_add("*Listbox.HighlightBackground", colors["border"])
        self.option_add("*Text.Background", colors["field"])
        self.option_add("*Text.Foreground", colors["text"])
        self.option_add("*Text.InsertBackground", colors["text"])
        self.option_add("*Text.SelectBackground", colors["selection"])
        self.option_add("*Text.SelectForeground", colors["text"])

        style.configure(".", background=colors["background"], foreground=colors["text"], fieldbackground=colors["field"])
        style.configure("TFrame", background=colors["background"])
        style.configure("Panel.TFrame", background=colors["panel"])
        style.configure("TLabel", background=colors["background"], foreground=colors["text"])
        style.configure("Muted.TLabel", background=colors["background"], foreground=colors["muted"])
        style.configure(
            "TLabelframe",
            background=colors["panel"],
            bordercolor=colors["border"],
            darkcolor=colors["border"],
            lightcolor=colors["border"],
        )
        style.configure("TLabelframe.Label", background=colors["panel"], foreground=colors["text"])
        style.configure(
            "TButton",
            background=colors["panel_alt"],
            foreground=colors["text"],
            bordercolor=colors["border"],
            focusthickness=1,
            focuscolor=colors["accent"],
            padding=(10, 6),
        )
        style.map(
            "TButton",
            background=[("active", colors["accent_active"]), ("pressed", colors["accent_active"])],
            foreground=[("disabled", colors["muted"])],
        )
        style.configure(
            "TEntry",
            fieldbackground=colors["field"],
            foreground=colors["text"],
            insertcolor=colors["text"],
            bordercolor=colors["border"],
            lightcolor=colors["border"],
            darkcolor=colors["border"],
        )
        style.map(
            "TEntry",
            fieldbackground=[("readonly", colors["field"]), ("disabled", colors["panel"])],
            foreground=[("disabled", colors["muted"])],
        )
        style.configure(
            "TCombobox",
            fieldbackground=colors["field"],
            background=colors["panel_alt"],
            foreground=colors["text"],
            arrowcolor=colors["text"],
            bordercolor=colors["border"],
            lightcolor=colors["border"],
            darkcolor=colors["border"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", colors["field"])],
            foreground=[("readonly", colors["text"]), ("disabled", colors["muted"])],
            selectbackground=[("readonly", colors["field"])],
            selectforeground=[("readonly", colors["text"])],
        )
        style.configure("TCheckbutton", background=colors["panel"], foreground=colors["text"])
        style.configure(
            "DangerBold.TLabel",
            background=colors["panel"],
            foreground=colors["danger"],
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TCheckbutton",
            background=[("active", colors["panel"])],
            foreground=[("disabled", colors["muted"])],
        )

    def style_native_widgets(self) -> None:
        colors = DARK_COLORS
        self.audio_list.configure(
            background=colors["field"],
            foreground=colors["text"],
            selectbackground=colors["selection"],
            selectforeground=colors["text"],
            highlightbackground=colors["border"],
            highlightcolor=colors["accent"],
            relief=tk.FLAT,
            borderwidth=1,
        )
        self.log.configure(
            background=colors["field"],
            foreground=colors["text"],
            insertbackground=colors["text"],
            selectbackground=colors["selection"],
            selectforeground=colors["text"],
            highlightbackground=colors["border"],
            highlightcolor=colors["accent"],
            relief=tk.FLAT,
            borderwidth=1,
        )

    def build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_NAME, font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status).grid(row=0, column=1, sticky="e")

        live_frame = ttk.LabelFrame(outer, text="Live Clipping")
        live_frame.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        for column in range(4):
            live_frame.columnconfigure(column, weight=1)

        ttk.Label(live_frame, text="Video source").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        source_combo = ttk.Combobox(
            live_frame,
            textvariable=self.video_source,
            values=("directshow", "screen", "camera"),
            state="readonly",
        )
        source_combo.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        ttk.Label(live_frame, text="Named video device").grid(row=0, column=1, columnspan=2, sticky="w", padx=8, pady=(8, 2))
        self.video_combo = ttk.Combobox(live_frame, textvariable=self.video_device, values=())
        self.video_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        ttk.Button(live_frame, text="Refresh Devices", command=self.refresh_all_devices).grid(
            row=1, column=3, sticky="ew", padx=8, pady=(0, 8)
        )

        ttk.Label(live_frame, text="Monitor / OBS projector").grid(row=2, column=0, sticky="w", padx=8, pady=(4, 2))
        self.monitor_combo = ttk.Combobox(live_frame, textvariable=self.monitor, values=(), state="readonly")
        self.monitor_combo.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))

        ttk.Label(live_frame, text="Clip seconds").grid(row=2, column=1, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(live_frame, textvariable=self.clip_seconds).grid(row=3, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Label(live_frame, text="Local-buffer clips only", style="Muted.TLabel").grid(
            row=4, column=1, sticky="w", padx=8, pady=(0, 2)
        )

        ttk.Label(live_frame, text="FPS").grid(row=2, column=2, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(live_frame, textvariable=self.fps).grid(row=3, column=2, sticky="ew", padx=8, pady=(0, 8))
        ttk.Label(live_frame, text="Local-buffer clips only", style="Muted.TLabel").grid(
            row=4, column=2, sticky="w", padx=8, pady=(0, 2)
        )

        ttk.Button(live_frame, text="Use OBS Projector Mode", command=self.use_obs_projector_mode).grid(
            row=3, column=3, sticky="ew", padx=8, pady=(0, 8)
        )

        ttk.Label(live_frame, text="Live command listener").grid(row=5, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Combobox(live_frame, textvariable=self.voice_provider, values=("vosk", "openai"), state="readonly").grid(
            row=6, column=0, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(live_frame, text="Clip action").grid(row=5, column=1, sticky="w", padx=8, pady=(4, 2))
        ttk.Combobox(live_frame, textvariable=self.clip_action, values=("obs_replay_buffer", "local_buffer"), state="readonly").grid(
            row=6, column=1, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Button(live_frame, text="Start Live Clipper", command=self.start_live_clipper).grid(
            row=6, column=2, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Button(live_frame, text="Clip Now", command=self.clip_now).grid(row=6, column=3, sticky="ew", padx=8, pady=(0, 8))

        ttk.Label(live_frame, text="Select one or more input devices to monitor").grid(
            row=7, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 2)
        )
        live_naming_frame = ttk.Frame(live_frame)
        live_naming_frame.grid(row=7, column=3, sticky="w", padx=8, pady=(4, 2))
        ttk.Checkbutton(
            live_naming_frame,
            text="Name live clips immediately",
            variable=self.name_live_clips,
        ).pack(anchor="w")
        ttk.Label(
            live_naming_frame,
            text="I also like to live dangerously",
            style="DangerBold.TLabel",
        ).pack(anchor="w")
        self.audio_list = tk.Listbox(live_frame, height=5, selectmode=tk.MULTIPLE, exportselection=False)
        self.audio_list.grid(row=8, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        audio_buttons = ttk.Frame(live_frame)
        audio_buttons.grid(row=8, column=3, sticky="nsew", padx=8, pady=(0, 8))
        ttk.Button(audio_buttons, text="Refresh Audio", command=self.refresh_audio_devices).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(audio_buttons, text="Use Line In + USB", command=self.use_default_audio_pair).pack(fill=tk.X)

        rename_frame = ttk.LabelFrame(outer, text="Renaming")
        rename_frame.grid(row=2, column=0, sticky="ew", pady=8)
        for column in range(5):
            rename_frame.columnconfigure(column, weight=1)
        ttk.Label(rename_frame, text="OBS recording or replay-buffer folder").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        ttk.Entry(rename_frame, textvariable=self.obs_output_dir).grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(rename_frame, text="Browse", command=self.browse_obs_folder).grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(rename_frame, text="Rename One Clip", command=self.rename_one_clip).grid(row=1, column=2, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(rename_frame, text="Batch Rename Existing", command=self.batch_rename_existing_clips).grid(
            row=1, column=3, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Button(rename_frame, text="Start OBS Renamer", command=self.start_obs_renamer).grid(
            row=1, column=4, sticky="ew", padx=8, pady=(0, 8)
        )

        rest_frame = ttk.LabelFrame(outer, text="Settings And Logs")
        rest_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        rest_frame.columnconfigure(0, weight=1)
        rest_frame.rowconfigure(2, weight=1)

        settings_frame = ttk.Frame(rest_frame)
        settings_frame.grid(row=0, column=0, sticky="ew")
        for column in range(4):
            settings_frame.columnconfigure(column, weight=1)

        ttk.Label(settings_frame, text="FFmpeg path").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        ttk.Entry(settings_frame, textvariable=self.ffmpeg_path).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(settings_frame, text="Live Vosk model folder").grid(row=0, column=2, sticky="w", padx=8, pady=(8, 2))
        ttk.Entry(settings_frame, textvariable=self.vosk_model_path).grid(
            row=1, column=2, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Button(settings_frame, text="Browse Live Vosk", command=self.browse_vosk_model).grid(
            row=1, column=3, sticky="ew", padx=8, pady=(0, 8)
        )

        ttk.Label(settings_frame, text="Rename Vosk model folder").grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(settings_frame, textvariable=self.rename_vosk_model_path).grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Button(settings_frame, text="Browse Rename Vosk", command=self.browse_rename_vosk_model).grid(
            row=3, column=3, sticky="ew", padx=8, pady=(0, 8)
        )

        ttk.Label(settings_frame, text="OBS host").grid(row=4, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(settings_frame, textvariable=self.obs_host).grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Label(settings_frame, text="OBS port").grid(row=4, column=1, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(settings_frame, textvariable=self.obs_port).grid(row=5, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Label(settings_frame, text="OBS password").grid(row=4, column=2, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(settings_frame, textvariable=self.obs_password, show="*").grid(row=5, column=2, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(settings_frame, text="Save OBS Replay", command=self.save_obs_replay).grid(
            row=5, column=3, sticky="ew", padx=8, pady=(0, 8)
        )

        ai_frame = ttk.Frame(rest_frame)
        ai_frame.grid(row=1, column=0, sticky="ew")
        for column in range(4):
            ai_frame.columnconfigure(column, weight=1)
        ttk.Label(ai_frame, text="Rename AI provider").grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 2))
        ttk.Combobox(ai_frame, textvariable=self.ai_provider, values=("lmstudio", "openai"), state="readonly").grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(ai_frame, text="Rename transcription").grid(row=0, column=2, columnspan=2, sticky="w", padx=8, pady=(8, 2))
        ttk.Combobox(
            ai_frame,
            textvariable=self.rename_transcription_provider,
            values=("local_whisper", "vosk", "openai", "disabled"),
            state="readonly",
        ).grid(row=1, column=2, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        self.local_whisper_fields = ttk.Frame(ai_frame)
        self.local_whisper_fields.grid(row=2, column=0, columnspan=4, sticky="ew")
        for column in range(4):
            self.local_whisper_fields.columnconfigure(column, weight=1)
        ttk.Label(self.local_whisper_fields, text="Whisper model").grid(row=0, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.local_whisper_fields, textvariable=self.local_whisper_model).grid(
            row=1, column=0, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.local_whisper_fields, text="Whisper device").grid(row=0, column=1, sticky="w", padx=8, pady=(4, 2))
        ttk.Combobox(
            self.local_whisper_fields,
            textvariable=self.local_whisper_device,
            values=("auto", "cuda", "cpu"),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Label(self.local_whisper_fields, text="Compute type").grid(row=0, column=2, sticky="w", padx=8, pady=(4, 2))
        ttk.Combobox(
            self.local_whisper_fields,
            textvariable=self.local_whisper_compute_type,
            values=("int8", "float16", "int8_float16", "float32"),
            state="readonly",
        ).grid(row=1, column=2, sticky="ew", padx=8, pady=(0, 8))
        ttk.Label(self.local_whisper_fields, text="CPU threads").grid(row=0, column=3, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.local_whisper_fields, textvariable=self.local_whisper_cpu_threads).grid(
            row=1, column=3, sticky="ew", padx=8, pady=(0, 8)
        )
        self.openai_fields = ttk.Frame(ai_frame)
        self.openai_fields.grid(row=3, column=0, columnspan=4, sticky="ew")
        for column in range(4):
            self.openai_fields.columnconfigure(column, weight=1)
        ttk.Label(self.openai_fields, text="API key env var").grid(row=0, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.openai_fields, textvariable=self.openai_key_env).grid(
            row=1, column=0, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.openai_fields, text="API key").grid(row=0, column=1, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.openai_fields, textvariable=self.openai_api_key, show="*").grid(
            row=1, column=1, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.openai_fields, text="Naming model").grid(row=0, column=2, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.openai_fields, textvariable=self.openai_naming_model).grid(
            row=1, column=2, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.openai_fields, text="Max frames").grid(row=0, column=3, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.openai_fields, textvariable=self.openai_max_frames).grid(
            row=1, column=3, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.openai_fields, text="Live command model").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 2)
        )
        ttk.Entry(self.openai_fields, textvariable=self.openai_voice_command_transcription_model).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.openai_fields, text="Rename transcript model").grid(
            row=2, column=2, columnspan=2, sticky="w", padx=8, pady=(4, 2)
        )
        ttk.Entry(self.openai_fields, textvariable=self.openai_rename_transcription_model).grid(
            row=3, column=2, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )

        self.lmstudio_fields = ttk.Frame(ai_frame)
        self.lmstudio_fields.grid(row=4, column=0, columnspan=4, sticky="ew")
        for column in range(4):
            self.lmstudio_fields.columnconfigure(column, weight=1)
        ttk.Label(self.lmstudio_fields, text="LM Studio base URL").grid(row=0, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.lmstudio_fields, textvariable=self.lmstudio_base_url).grid(
            row=1, column=0, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.lmstudio_fields, text="LM Studio naming model").grid(row=0, column=1, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.lmstudio_fields, textvariable=self.lmstudio_model).grid(
            row=1, column=1, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.lmstudio_fields, text="Token env var").grid(row=0, column=2, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.lmstudio_fields, textvariable=self.lmstudio_key_env).grid(
            row=1, column=2, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(self.lmstudio_fields, text="API key").grid(row=0, column=3, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(self.lmstudio_fields, textvariable=self.lmstudio_api_key, show="*").grid(
            row=1, column=3, sticky="ew", padx=8, pady=(0, 8)
        )

        ttk.Checkbutton(
            ai_frame,
            text="Enable OBS scene/source voice switching",
            variable=self.enable_obs_scene_source_switching,
        ).grid(row=5, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 8))
        ttk.Label(ai_frame, text="Filename prefix").grid(row=6, column=0, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(ai_frame, textvariable=self.filename_prefix).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(ai_frame, text="Filename suffix").grid(row=6, column=2, sticky="w", padx=8, pady=(4, 2))
        ttk.Entry(ai_frame, textvariable=self.filename_suffix).grid(
            row=7, column=2, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )

        action_frame = ttk.Frame(rest_frame)
        action_frame.grid(row=2, column=0, sticky="nsew")
        action_frame.columnconfigure(0, weight=1)
        action_frame.rowconfigure(1, weight=1)

        buttons = ttk.Frame(action_frame)
        buttons.grid(row=0, column=0, sticky="ew")
        for column in range(4):
            buttons.columnconfigure(column, weight=1)
        ttk.Button(buttons, text="Save Config", command=self.save_from_ui).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Stop", command=self.stop_process).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(buttons, text="Open Clips Folder", command=self.open_clips_folder).grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(buttons, text="Contact Me", command=self.open_contact_link).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        self.log = tk.Text(action_frame, height=12, wrap=tk.WORD)
        self.log.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.log.configure(state=tk.DISABLED)
        self.style_native_widgets()

    def update_ai_provider_fields(self) -> None:
        if not hasattr(self, "openai_fields") or not hasattr(self, "lmstudio_fields"):
            return
        ai_provider = self.ai_provider.get().lower()
        voice_provider = self.voice_provider.get().lower()
        rename_transcription_provider = self.rename_transcription_provider.get().lower()
        if ai_provider == "openai" or voice_provider == "openai" or rename_transcription_provider == "openai":
            self.openai_fields.grid()
        else:
            self.openai_fields.grid_remove()

        if ai_provider == "lmstudio":
            self.lmstudio_fields.grid()
        else:
            self.lmstudio_fields.grid_remove()

        if rename_transcription_provider in {"local_whisper", "whisper", "faster_whisper"}:
            self.local_whisper_fields.grid()
        else:
            self.local_whisper_fields.grid_remove()

    def load_config(self) -> dict:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
                loaded = json.load(config_file)
            merged = DEFAULT_CONFIG.copy()
            merged.update(loaded)
            if "openai" in loaded:
                openai = DEFAULT_CONFIG["openai"].copy()
                openai.update(loaded["openai"])
                merged["openai"] = openai
            if "lmstudio" in loaded:
                lmstudio = DEFAULT_CONFIG["lmstudio"].copy()
                lmstudio.update(loaded["lmstudio"])
                merged["lmstudio"] = lmstudio
            if "local_whisper" in loaded:
                local_whisper = DEFAULT_CONFIG["local_whisper"].copy()
                local_whisper.update(loaded["local_whisper"])
                merged["local_whisper"] = local_whisper
            if "voice" in loaded:
                voice = DEFAULT_CONFIG["voice"].copy()
                voice.update(loaded["voice"])
                merged["voice"] = voice
            if "obs" in loaded:
                obs = DEFAULT_CONFIG["obs"].copy()
                obs.update(loaded["obs"])
                merged["obs"] = obs
            if str(merged.get("voice_command_provider", "")).lower() == "lmstudio":
                merged["voice_command_provider"] = "vosk"
                merged["voice"]["provider"] = "vosk"
            if str(merged.get("rename_transcription_provider", "")).lower() == "lmstudio":
                merged["rename_transcription_provider"] = "openai"
            return merged
        return DEFAULT_CONFIG.copy()

    def save_from_ui(self) -> None:
        try:
            self.config["video_source"] = self.video_source.get()
            self.config["video_device_name"] = self.video_device.get() or None
            self.config["monitor_index"] = self.selected_monitor_index()
            self.config["audio_device_name"] = None
            self.config["audio_device"] = None
            self.config["audio_devices"] = self.selected_audio_indexes()
            self.config["obs_output_dir"] = self.obs_output_dir.get() or None
            self.config["clip_seconds"] = int(self.clip_seconds.get())
            self.config["fps"] = int(self.fps.get())
            self.config["ffmpeg_path"] = self.ffmpeg_path.get() or "ffmpeg"
            self.config["ai_provider"] = self.ai_provider.get()
            self.config["rename_transcription_provider"] = self.rename_transcription_provider.get()
            self.config["name_live_clips"] = self.name_live_clips.get()
            self.config["filename_prefix"] = self.filename_prefix.get()
            self.config["filename_suffix"] = self.filename_suffix.get()
            self.config["openai"] = {
                "api_key": self.openai_api_key.get() or None,
                "api_key_env": self.openai_key_env.get() or "OPENAI_API_KEY",
                "voice_command_transcription_model": (
                    self.openai_voice_command_transcription_model.get() or "gpt-4o-mini-transcribe"
                ),
                "rename_transcription_model": self.openai_rename_transcription_model.get() or "gpt-4o-mini-transcribe",
                "naming_model": self.openai_naming_model.get() or "gpt-4.1-mini",
                "max_frames_for_naming": int(self.openai_max_frames.get() or "8"),
            }
            self.config["lmstudio"] = {
                "base_url": self.lmstudio_base_url.get() or "http://localhost:1234/v1",
                "api_key": self.lmstudio_api_key.get() or None,
                "api_key_env": self.lmstudio_key_env.get() or "LMSTUDIO_API_KEY",
                "vision_model": self.lmstudio_model.get() or "qwen2.5-vl-7b-instruct",
            }
            self.config["local_whisper"] = {
                "model_size": self.local_whisper_model.get() or "base.en",
                "device": self.local_whisper_device.get() or "auto",
                "compute_type": self.local_whisper_compute_type.get() or "int8",
                "cpu_threads": int(self.local_whisper_cpu_threads.get() or "0"),
            }
            self.config["voice_command_provider"] = self.voice_provider.get()
            self.config["voice"] = {
                "provider": self.voice_provider.get(),
                "vosk_model_path": self.vosk_model_path.get() or "models/vosk-model-en-us-0.22-lgraph",
                "rename_vosk_model_path": self.rename_vosk_model_path.get() or "models/vosk-model-en-us-0.22-lgraph",
                "trigger_cooldown_seconds": 2,
                "clip_action": self.clip_action.get(),
                "enable_obs_scene_source_switching": self.enable_obs_scene_source_switching.get(),
            }
            self.config["obs"] = {
                "host": self.obs_host.get() or "localhost",
                "port": int(self.obs_port.get() or "4455"),
                "password_env": "OBS_WEBSOCKET_PASSWORD",
                "request_timeout_seconds": 5,
            }
        except ValueError as exc:
            messagebox.showerror("Invalid Config", str(exc))
            return

        with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
            json.dump(self.config, config_file, indent=2)
            config_file.write("\n")
        self.append_log(f"Saved {CONFIG_PATH.name}")

    def selected_audio_indexes(self) -> list[int]:
        indexes: list[int] = []
        for selected in self.audio_list.curselection():
            device_index, _name = self.audio_devices[selected]
            indexes.append(device_index)
        return indexes

    def refresh_all_devices(self) -> None:
        self.refresh_monitors()
        self.refresh_audio_devices()
        self.refresh_video_devices()

    def refresh_monitors(self) -> None:
        self.monitors.clear()
        try:
            with mss.mss() as sct:
                for index, monitor in enumerate(sct.monitors):
                    if index == 0:
                        label = f"0: All monitors ({monitor['width']}x{monitor['height']})"
                    else:
                        label = (
                            f"{index}: Monitor {index} "
                            f"{monitor['width']}x{monitor['height']} at {monitor['left']},{monitor['top']}"
                        )
                    self.monitors.append((index, label))
            self.monitor_combo.configure(values=[label for _index, label in self.monitors])
            configured = int(self.config.get("monitor_index", 1))
            for index, label in self.monitors:
                if index == configured:
                    self.monitor.set(label)
                    break
            if not self.monitor.get() and self.monitors:
                self.monitor.set(self.monitors[min(1, len(self.monitors) - 1)][1])
        except Exception as exc:
            self.append_log(f"Monitor refresh failed: {exc}")

    def selected_monitor_index(self) -> int:
        selected = self.monitor.get()
        for index, label in self.monitors:
            if label == selected:
                return index
        return int(self.config.get("monitor_index", 1))

    def use_obs_projector_mode(self) -> None:
        self.video_source.set("screen")
        self.append_log("OBS projector mode selected. Put OBS Fullscreen/Windowed Projector on the selected monitor.")

    def refresh_audio_devices(self) -> None:
        self.audio_devices.clear()
        self.audio_list.delete(0, tk.END)
        try:
            for index, device in enumerate(sd.query_devices()):
                if int(device.get("max_input_channels", 0)) <= 0:
                    continue
                name = str(device.get("name", "Unknown"))
                hostapi = sd.query_hostapis(int(device.get("hostapi", 0))).get("name", "")
                label = f"{index}: {name} ({hostapi})"
                self.audio_devices.append((index, label))
                self.audio_list.insert(tk.END, label)
            self.load_audio_selection()
        except Exception as exc:
            self.append_log(f"Audio device refresh failed: {exc}")

    def refresh_video_devices(self) -> None:
        ffmpeg = self.ffmpeg_path.get() or "ffmpeg"
        devices: list[str] = []
        command = [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        try:
            result = subprocess.run(command, cwd=APP_DIR, capture_output=True, text=True, timeout=20)
            text = result.stderr or result.stdout
            for line in text.splitlines():
                match = re.search(r'"([^"]+)" \(video\)', line)
                if match:
                    devices.append(match.group(1))
        except Exception as exc:
            self.append_log(f"Video device refresh failed: {exc}")
        self.video_devices = devices
        self.video_combo.configure(values=devices)
        if devices and not self.video_device.get():
            self.video_device.set(devices[0])
        self.append_log(f"Found {len(devices)} named video devices")

    def load_audio_selection(self) -> None:
        configured = set(self.config.get("audio_devices") or [])
        for list_index, (device_index, label) in enumerate(self.audio_devices):
            if device_index in configured or str(device_index) in configured:
                self.audio_list.selection_set(list_index)
            elif "Line In (Realtek(R) Audio)" in label or "Line (2- USB AUDIO  CODEC)" in label:
                if not configured:
                    self.audio_list.selection_set(list_index)

    def use_default_audio_pair(self) -> None:
        self.audio_list.selection_clear(0, tk.END)
        wanted = ("Line In (Realtek(R) Audio)", "Line (2- USB AUDIO  CODEC)")
        for list_index, (_device_index, label) in enumerate(self.audio_devices):
            if any(name in label for name in wanted):
                self.audio_list.selection_set(list_index)

    def browse_obs_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.obs_output_dir.get() or str(APP_DIR))
        if folder:
            self.obs_output_dir.set(folder)

    def browse_vosk_model(self) -> None:
        folder = filedialog.askdirectory(initialdir=str(APP_DIR / "models"))
        if folder:
            try:
                self.vosk_model_path.set(str(Path(folder).resolve().relative_to(APP_DIR)))
            except ValueError:
                self.vosk_model_path.set(folder)

    def browse_rename_vosk_model(self) -> None:
        folder = filedialog.askdirectory(initialdir=str(APP_DIR / "models"))
        if folder:
            try:
                self.rename_vosk_model_path.set(str(Path(folder).resolve().relative_to(APP_DIR)))
            except ValueError:
                self.rename_vosk_model_path.set(folder)

    def rename_one_clip(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=self.obs_output_dir.get() or str(APP_DIR),
            filetypes=(("Video files", "*.mp4 *.mkv *.mov *.flv"), ("All files", "*.*")),
        )
        if path:
            self.obs_output_dir.set(str(Path(path).parent))
            self.save_from_ui()
            self.start_worker_process(["--rename-file", path])

    def start_live_clipper(self) -> None:
        self.save_from_ui()
        self.start_worker_process(["--live-clipper"])

    def clip_now(self) -> None:
        if not self.process or self.process.poll() is not None:
            messagebox.showinfo("Live Clipper Not Running", "Start the live clipper before using Clip Now.")
            return
        buffer_dir = APP_DIR / str(self.config.get("buffer_dir", ".capture_buffer"))
        buffer_dir.mkdir(parents=True, exist_ok=True)
        trigger_path = buffer_dir / "clip_now.trigger"
        trigger_path.write_text("clip\n", encoding="utf-8")
        self.append_log("Clip Now requested")

    def save_obs_replay(self) -> None:
        self.save_from_ui()
        if self.process and self.process.poll() is None:
            self.clip_now()
            return
        self.start_worker_process(["--save-obs-replay-buffer"])

    def start_obs_renamer(self) -> None:
        self.save_from_ui()
        if not self.obs_output_dir.get():
            messagebox.showerror("OBS Folder Missing", "Choose your OBS recording or replay-buffer folder first.")
            return
        self.start_worker_process(["--watch-obs-clips"])

    def batch_rename_existing_clips(self) -> None:
        self.save_from_ui()
        if not self.obs_output_dir.get():
            messagebox.showerror("OBS Folder Missing", "Choose your OBS recording or replay-buffer folder first.")
            return
        self.append_log("Batch rename requested for existing OBS clips.")
        self.start_worker_process(["--batch-rename-obs-clips"])

    def worker_command(self, args: list[str]) -> list[str]:
        return build_worker_command(args)

    def start_worker_process(self, args: list[str]) -> None:
        self.start_process(self.worker_command(args))

    def start_process(self, command: list[str]) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showinfo("Already Running", "Stop the current process before starting another one.")
            return
        self.append_log(f"Starting: {' '.join(command)}")
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self.process = subprocess.Popen(
            command,
            cwd=APP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
            env=self.child_env(),
        )
        self.status.set("Running")
        threading.Thread(target=self.read_process_output, daemon=True).start()

    def child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.obs_password.get():
            env["OBS_WEBSOCKET_PASSWORD"] = self.obs_password.get()
        return env

    def read_process_output(self) -> None:
        process = self.process
        if not process or not process.stdout:
            return
        for line in process.stdout:
            self.after(0, self.append_log, line.rstrip())
        code = process.wait()
        self.after(0, self.status.set, f"Stopped ({code})")
        self.after(0, self.append_log, f"Process exited with code {code}")

    def stop_process(self) -> None:
        if self.process and self.process.poll() is None:
            self.append_log("Stopping process...")
            self.process.terminate()
            self.status.set("Stopping")
        else:
            self.status.set("Idle")

    def open_clips_folder(self) -> None:
        folder = APP_DIR / str(self.config.get("output_dir", "clips"))
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def open_contact_link(self) -> None:
        webbrowser.open(REPO_URL)

    def append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Process Running", "Stop the running process and close?"):
                return
            self.stop_process()
        self.destroy()


def build_worker_command(args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    python_path = APP_DIR / ".venv" / "Scripts" / "python.exe"
    python_exe = str(python_path) if python_path.exists() else sys.executable
    return [python_exe, "-u", str(SCRIPT_PATH), *args]


def print_ui_action_audit() -> None:
    print("UI action audit")
    print("Local UI-only actions:")
    print("  Refresh Devices, Use OBS Projector Mode, Refresh Audio, Use Line In + USB")
    print("  Browse, Browse Vosk Model, Save Config, Clip Now, Stop, Open Clips Folder, Contact Me")
    print("Worker actions:")
    for label, args in WORKER_ARG_SETS.items():
        print(f"  {label}: {' '.join(build_worker_command(args))}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--ui-action-audit"]:
        print_ui_action_audit()
        raise SystemExit(0)
    if len(sys.argv) > 1:
        from live_video_interpreter import main

        main()
    else:
        ControlPanel().mainloop()
