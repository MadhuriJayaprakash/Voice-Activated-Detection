# AI Speech Analyzer

Speech transcription with speaker diarization and silence detection.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your HuggingFace token to `.env`:
```bash
# Edit .env file
HF_TOKEN=your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

Accept model terms:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

3. Start the app:
```bash
python app.py
```

4. Open browser: http://localhost:5000

## Usage

- Models auto-load on startup (takes 2-4 minutes)
- Select Whisper model size in UI
- Upload audio file
- Get transcript with speaker labels

## Features

- Speech transcription (Whisper)
- Speaker diarization (PyAnnote)
- Silence detection
- Silent audio detection (shows 0 speakers correctly)
- Download results as CSV, JSON, or RTTM
