
Are you a lazy AF "tech influcener"? 

Do you farm clips by giving obvious tech tips and "hot takes"? 

Do you think Python is a snek that lives in Asia? 

If so, you probably shouldn't be giving people tech advice.. yet here we are. Today is your lucky day!
```

 ▄█ ▄ █▀█  ▄▀▀▀▀█  ▄█      ▄▀▀▀▀█  ▄▀▀▀▀█  ▄▀█▄  ▄█▀█  ▄▀▀▀▀█       ▄▀▀▀▀▀█  ▄▀▀▀▀█     
█ ██ ██ █ ▓ ▄▀▀▀▀ ▓ █     ▓ ▄▀▀▀▀ ▓ ▄▀█ █ ▓ ▄ █ █   █ ▓ ▄▀▀▀▀      ▀▀▀█ █▀▀ ▓ ▄▀█ █     
█ █ █ █ █ ▒ ▄▀▀   ▒ ▄     ▒ ▄     ▒ █ █ █ ▒ ██ ▀ █▄ █ ▒ ▄▀▀           █ █   ▒ █ █ █     
█  █ █  █ ░ █▄▄▄▄ ░ █▄▄▄▄ ░ █▄▄▄▄ ░ █▄▀ █ ░ █ ▀▀▀ █ █ ░ █▄▄▄▄         █ █   ░ █▄▀ █     
█▄█   █▀  █▄▄▄▄▀  █▄▄▄▄▀  █▄▄▄▄▀  █▄▄▄▄▀  █▄█     █▀  █▄▄▄▄▀          █▀    █▄▄▄▄▀      
 ▄▀▀▀▀█  ▄█      ▄█  ▄▀▀▀▀▓       ▄▀█▄  ▄█▀█  ▄▀▀▀▀█  ▄▀▀▀▀█  ▄▀▀▀▀▀█  ▄▀▀▀▀█  ▄▀▀▀▀▓  
▓ ▄▀▀▀▀ ▓ █     █ █ ▓ ▄▀█ ▒      ▓ ▄ █ █   █ ▓ ▄▀█ █ █ ▄▀▀▀▀ ▀▀▀█ █▀▀ ▓ ▄▀▀▀▀ ▓ ▄▀█ ▒  
▒ ▄     ▒ ▄     █ █ ▒ ▄▀▀▀       ▒ ██ ▀ █▄ █ ▒ ▀▀▀ █ ▀▄▄█▀▀▄    █ █   ▒ ▄▀▀   ▒ ▄▀▀░   
░ █▄▄▄▄ ░ █▄▄▄▄ █ █ ░ █          ░ █ ▀▀▀ █ █ ░ █▀█ █ ▄▄▄▄▀ █    █ █   ░ █▄▄▄▄ ░ █▀▄ ▀▄ 
█▄▄▄▄▀  █▄▄▄▄▀  █▀  █▄█          █▄█     █▀  █▄█ █▀  █▄▄▄▄▀     █▀    █▄▄▄▄▀  █▄█  ▀▄▄█

```

A largely inefficent, slightly buggy work of jankery. It basically interprets voice commands from the actual mic you use to stream on. You can start/end replay buffer, save replay buffer, and switch scenes all through voice intent. It also uses a vision model plus optional rename-time audio transcription to analyze your clips
and recommend new file names so you can keep track of them. I currently have it setup to use LM Studio or OpenAI for clip naming, OpenAI for rename-time transcription, and Vosk/OpenAI for live voice commands. It can be easily retooled for any AI model run in various fashions. 

## What It Does

- Saves the active OBS replay buffer when you say `clip that`.
- Starts and stops the OBS replay buffer by voice.
- Starts and stops OBS recording by voice.
- Optionally switches OBS scenes or sources by voice.
- Saves local live-buffer clips with default timestamp names by default.
- Watches an OBS output folder and renames clips with AI-generated names.

## Requirements

- Windows
- Python 3.10 or newer
- OBS Studio with WebSocket enabled
- A microphone or audio input device
- Optional: FFmpeg on `PATH` for local-buffer capture mode

OBS WebSocket is built into modern OBS Studio. In OBS, check:

```text
Tools -> WebSocket Server Settings
```

Use the same host, port, and password in `config.json`.

## Install

From PowerShell:

```powershell
git clone https://github.com/ISuckatCod3/Clip-Master-9000.git
cd Clip-Master-9000
.\build.bat
```

One-line install from PowerShell is also supported:

```powershell
irm https://raw.githubusercontent.com/ISuckatCod3/Clip-Master-9000/main/install.ps1 | iex
```

That installs to `%LOCALAPPDATA%\Programs\Clip-Master-9000`, runs the same build checks, and creates desktop shortcuts for the UI and listener.

`build.bat` runs the full local setup and verification sequence:

- checks Python 3.10+
- creates `.venv`
- installs dependencies, including Vosk
- creates `config.json`
- downloads and extracts the default Vosk model
- creates runtime folders
- validates Python files and JSON config
- checks required package imports

Manual setup is also available:

```powershell
.\setup.bat
```

`setup.bat` will:

- create `.venv`
- install Python dependencies, including Vosk
- create `config.json` from `config.example.json`
- download and extract the larger English Vosk voice model

The Vosk model used by default is `vosk-model-en-us-0.22-lgraph`, listed on the official Vosk model page. It is larger than the tiny/small model and is bundled for more reliable command recognition.

## Run

UI:

```powershell
.\run_ui.bat
```

Headless voice listener:

```powershell
.\run_listener.bat
```

OBS clip renamer:

```powershell
.\run_obs_renamer.bat
```

Batch rename existing OBS clips:

```powershell
.\.venv\Scripts\python.exe .\live_video_interpreter.py --batch-rename-obs-clips
```

Batch rename a specific folder without changing `config.json`:

```powershell
.\.venv\Scripts\python.exe .\live_video_interpreter.py --batch-rename-obs-clips --rename-folder "D:\OBS\Replays"
```

Rename one clip directly:

```powershell
.\.venv\Scripts\python.exe .\live_video_interpreter.py --rename-file "D:\OBS\Replays\clip.mp4"
```

Live clipping is uncoupled from AI naming by default: local-buffer clips save immediately with default timestamp names such as `clip_2026-06-12_12-34-56.mp4`. Turn on `Name live clips immediately` in the UI only if you want live clips renamed as they are created.

The renamer asks the configured vision model to read visible RTSS/MSI Afterburner-style overlay details and include clear PC specs in generated names. Filename prefix and suffix fields are available in the UI and apply to AI naming flows: one-off rename, live watch, batch rename, and optional immediate live-clip naming.

Save replay buffer once:

```powershell
.\save_obs_replay.bat
```

## Package

Portable zip:

```powershell
.\package.ps1 -Target Portable
```

PyInstaller app folder:

```powershell
.\package.ps1 -Target Exe
```

The portable package is the most reliable distribution format. MSI can be built from the portable folder with WiX Toolset when a formal installer is needed.

## Logs

Launch scripts write logs under `logs\`.

Useful files:

- `logs\listener.out.log`
- `logs\listener.err.log`
- `logs\ui.out.log`
- `logs\ui.err.log`
- `logs\obs_renamer.out.log`
- `logs\obs_renamer.err.log`



## Config

Edit `config.json`.

Important fields:

```json
"name_live_clips": false,
"obs": {
  "host": "localhost",
  "port": 4455,
  "password": null,
  "password_env": "OBS_WEBSOCKET_PASSWORD",
  "request_timeout_seconds": 5
}
```

Voice settings:

```json
"voice_command_provider": "vosk",
"rename_transcription_provider": "local_whisper",
"local_whisper": {
  "model_size": "base.en",
  "device": "auto",
  "compute_type": "int8",
  "cpu_threads": 0
},
"voice": {
  "provider": "vosk",
  "vosk_model_path": "models/vosk-model-en-us-0.22-lgraph",
  "rename_vosk_model_path": "models/vosk-model-en-us-0.22-lgraph",
  "trigger_cooldown_seconds": 2,
  "clip_action": "obs_replay_buffer",
  "enable_obs_scene_source_switching": false
}
```

Vosk is the default live voice command engine. A normal `setup.bat` run installs the Python dependencies, including `vosk` and `faster-whisper`, and downloads the default Vosk model into:

```text
models\vosk-model-en-us-0.22-lgraph
```

Use `setup.ps1 -SkipVoskModel` if you plan to provide your own model path in `config.json`.

Test command matching without using the microphone, OBS, or capture devices:

```powershell
.\.venv\Scripts\python.exe -u .\live_video_interpreter.py --voice-command-smoke-test "clippy clip that"
```

Matched phrases print the action that would run and exit with code `0`. Non-command text prints `No voice command matched` and exits with code `1`.

Live voice commands and clip renaming transcription are separate settings:

- `voice_command_provider` controls live spoken commands such as `clip that`; use `vosk` for local low-latency command listening or `openai` for API transcription.
- `voice.require_wake_phrase` keeps commands gated behind `clippy` or `clip master` by default. Say a wake phrase, then speak a command within `voice.wake_listen_seconds`.
- `rename_transcription_provider` controls whether clip audio is transcribed before AI naming; use `local_whisper` for faster local transcript-aware names, `vosk` for the large Vosk model, `openai` for API transcription, or `disabled` to name from frames only.
- `local_whisper.model_size` can be a faster-whisper model name such as `base.en`, `small.en`, or a local model path. The first use may download the model.
- `voice.rename_vosk_model_path` should point at the large Vosk model if `rename_transcription_provider` is set to `vosk`. This is separate from `voice.vosk_model_path`, so live commands can use a faster model later without downgrading Vosk rename transcripts.
- `openai.voice_command_transcription_model` is used only for OpenAI-powered live commands.
- `openai.rename_transcription_model` is used only for transcript context during one-off, batch, live-watch, and optional immediate live-clip naming.

## LM Studio Setup

LM Studio is optional and can be used for local AI clip naming through its OpenAI-compatible server. It is not used for live voice commands or audio transcription in this app. Live voice commands use Vosk by default, or OpenAI transcription if selected; rename-time audio transcription uses local Whisper by default, Vosk when `rename_transcription_provider` is set to `vosk`, or OpenAI when it is set to `openai`.

1. Install LM Studio.
2. Download a vision-capable model for clip naming.
3. Start the LM Studio local server.
4. Confirm the server URL is:

```text
http://localhost:1234/v1
```

5. Set `ai_provider` to `lmstudio` in `config.json` if LM Studio should name clips.
6. Set `lmstudio.vision_model` to the model you loaded in LM Studio.

The LM Studio API key can stay blank for the local server; the app supplies a local placeholder automatically.

Recommended models:

- `qwen/qwen2.5-vl-7b` for higher-performance 12GB+ GPUs.
- `qwen3-vl-2b` for lower-end GPUs.

Example:

```json
"ai_provider": "lmstudio",
"lmstudio": {
  "base_url": "http://localhost:1234/v1",
  "api_key": null,
  "api_key_env": "LMSTUDIO_API_KEY",
  "vision_model": "qwen3-vl-2b"
}
```

Set OBS password either directly:

```json
"password": "your-password"
```

or with an environment variable:

```powershell
setx OBS_WEBSOCKET_PASSWORD "your-password"
```

## Voice Commands

Built in:

- `clippy`, then one of the commands below
- `clip master`, then one of the commands below
- `clip that`
- `save that`
- `record that`
- `capture that`
- `start replay buffer`
- `stop replay buffer`
- `start recording`
- `stop recording`

OBS WebSocket request names used by this app:

| OBS WebSocket request | Recommended voice phrase |
| --- | --- |
| `GetReplayBufferStatus` | `clippy start replay buffer` or `clippy stop replay buffer` |
| `StartReplayBuffer` | `clippy start replay buffer` |
| `StopReplayBuffer` | `clippy stop replay buffer` |
| `SaveReplayBuffer` | Use the UI `Save OBS Replay` button or CLI helper. |
| `GetRecordStatus` | `clippy start recording` or `clippy stop recording` |
| `StartRecord` | `clippy start recording` |
| `StopRecord` | `clippy stop recording` |
| `GetSceneList` | `clippy switch to Gameplay` |
| `GetInputList` | `clippy source Main Camera` |
| `GetSceneItemList` | `clippy source Main Camera` |
| `SetCurrentProgramScene` | `clippy switch to Gameplay` |
| `SetSceneItemEnabled` | `clippy source Main Camera` |

Optional OBS scene/source switching is off by default. Enable it with:

```json
"enable_obs_scene_source_switching": true
```

Then use explicit commands:

- `switch to Gameplay`
- `go to BRB`
- `show Camera`
- `scene Intro`
- `source Elgato`

`Elgato` is only an example source name. The app reads your actual OBS scene names and scene-item source names through WebSocket, then matches the spoken target against those names. If a name is ambiguous, it refuses to switch instead of guessing.

For best voice matching, use proper names for OBS scenes and sources. Prefer names that sound like objects or layouts, such as `Gameplay`, `Starting Soon`, `Main Camera`, `Desk Mic`, or `Capture Card`. Avoid naming scenes or sources after command words or WebSocket request words, because those names can make the spoken intent ambiguous.

Avoid scene/source names like:

- `Start Recording`
- `Stop Replay Buffer`
- `Save Replay Buffer`
- `Switch Scene`
- `Get Scene List`
- `Set Current Program Scene`
- `Set Scene Item Enabled`
- `Source`
- `Scene`

Better names:

- `Gameplay`
- `Starting Soon`
- `BRB`
- `Main Camera`
- `Face Cam`
- `Capture Card`
- `Overlay Clean`
- `Overlay Full`

## CLI Helpers

```powershell
.\.venv\Scripts\python.exe .\live_video_interpreter.py --list-audio-devices
.\.venv\Scripts\python.exe .\live_video_interpreter.py --list-capture-devices
.\.venv\Scripts\python.exe .\live_video_interpreter.py --set-audio-devices 2 5
.\.venv\Scripts\python.exe .\live_video_interpreter.py --list-obs-scenes-sources
.\.venv\Scripts\python.exe .\live_video_interpreter.py --switch-obs-to "Gameplay"
```
## How to install for dummies
1. Clone the repo
2. Navigate to the project folder
3. Open Powershell with Admin
4. CD .\Clip-Master-9000
5. Run .\package.ps1 -Target exe
6. Open Explorer
7. Navigate to .\Clip-Master-9000\Dist\Clip Master 9000
8. Double-click Clip Master 9000.exe

UI will launch.

Set parameters and hit "Save Config"

## First Run Checklist *not needed if using .exe*

1. Run `setup.bat`.
2. Start OBS.
3. Enable OBS WebSocket.
4. Edit `config.json` if OBS is not on `localhost:4455`.
5. Run `run_ui.bat`.
6. Pick audio devices in the UI and save config.
7. Run `run_listener.bat`.
