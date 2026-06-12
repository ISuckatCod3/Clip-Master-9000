
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

 A largely inefficent, slightly buggy work of jankery. It basically interprets voice commands from the actual mic you use to stream on. You can start/end replay buffer, save replay buffer, and switch scenes all through voice intent. It also uses the VL + transcription model of your choice to analyze your clips
 and recommend new file names so you can keep track of them. I currently have it setup to use LM Studio MCP server or OpenAI API. It can be easily retooled for any AI model run in various fashions. 

## What It Does

- Saves the active OBS replay buffer when you say `clip that`.
- Starts and stops the OBS replay buffer by voice.
- Starts and stops OBS recording by voice.
- Optionally switches OBS scenes or sources by voice.
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
- download and extract the small English Vosk voice model

The Vosk model used by default is `vosk-model-small-en-us-0.15`, listed on the official Vosk model page.

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

The renamer asks the configured vision model to read visible RTSS/MSI Afterburner-style overlay details and include clear PC specs in generated names. Filename prefix and suffix fields are available in the UI and apply to one-off, live watch, and batch renames.

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
"voice": {
  "provider": "vosk",
  "vosk_model_path": "models/vosk-model-small-en-us-0.15",
  "trigger_cooldown_seconds": 2,
  "clip_action": "obs_replay_buffer",
  "enable_obs_scene_source_switching": false
}
```

Vosk is the default voice command engine. A normal `setup.bat` run installs the `vosk` Python package and downloads the default model into:

```text
models\vosk-model-small-en-us-0.15
```

Use `setup.ps1 -SkipVoskModel` if you plan to provide your own model path in `config.json`.

Test command matching without using the microphone, OBS, or capture devices:

```powershell
.\.venv\Scripts\python.exe -u .\live_video_interpreter.py --voice-command-smoke-test "clip that"
```

Matched phrases print the action that would run and exit with code `0`. Non-command text prints `No voice command matched` and exits with code `1`.

## LM Studio Setup

LM Studio is optional and is used for local AI clip naming. The voice transcription is still handled by Vosk.

1. Install LM Studio.
2. Download a vision-capable Qwen model.
3. Start the LM Studio local server.
4. Confirm the server URL is:

```text
http://localhost:1234/v1
```

5. Set `ai_provider` to `lmstudio` in `config.json`.
6. Set `lmstudio.vision_model` to the model you loaded in LM Studio.

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

- `clip that`
- `save that`
- `record that`
- `capture that`
- `start replay buffer`
- `stop replay buffer`
- `start recording`
- `stop recording`

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

The app reads OBS scene names and scene-item source names through WebSocket. If a name is ambiguous, it refuses to switch instead of guessing.

## CLI Helpers

```powershell
.\.venv\Scripts\python.exe .\live_video_interpreter.py --list-audio-devices
.\.venv\Scripts\python.exe .\live_video_interpreter.py --list-capture-devices
.\.venv\Scripts\python.exe .\live_video_interpreter.py --set-audio-devices 2 5
.\.venv\Scripts\python.exe .\live_video_interpreter.py --list-obs-scenes-sources
.\.venv\Scripts\python.exe .\live_video_interpreter.py --switch-obs-to "Gameplay"
```

## First Run Checklist

1. Run `setup.bat`.
2. Start OBS.
3. Enable OBS WebSocket.
4. Edit `config.json` if OBS is not on `localhost:4455`.
5. Run `run_ui.bat`.
6. Pick audio devices in the UI and save config.
7. Run `run_listener.bat`.
