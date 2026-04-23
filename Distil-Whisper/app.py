import sys, types
# Stub out k2_fsa before speechbrain lazy-loader triggers it
for _mod in ['k2', 'speechbrain.integrations.k2_fsa']:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
import huggingface_hub
import os
import json
import tempfile
from datetime import timedelta
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import gc
import warnings
import threading
import time
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'distil-whisper-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024   # 500 MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}})

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

# Available Distil-Whisper model variants
DISTIL_MODELS = {
    "distil-small.en":    {"label": "Distil-Small  (~166 MB, English only, fastest)", "hf_id": "distil-whisper/distil-small.en"},
    "distil-medium.en":   {"label": "Distil-Medium (~394 MB, English only, balanced)", "hf_id": "distil-whisper/distil-medium.en"},
    "distil-large-v3":    {"label": "Distil-Large-v3 (~756 MB, multilingual, best)", "hf_id": "distil-whisper/distil-large-v3"},
}


# ─────────────────────────────────────────────────────────────────────────────
class SpeechAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_pipe = None          # Distil-Whisper HuggingFace pipeline
        self.diarization_pipeline = None
        self.vad_pipeline = None
        self.models_loaded = False
        self.loading_progress = 0
        self.loading_status = "Not started"
        self.current_model_id = None
        self.timing = {}

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_models_async(self, hf_token, distil_model_key="distil-small.en"):
        def _check_cached(repo_id):
            """Return True if model is already in the HF cache (no network needed)."""
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id=repo_id, local_files_only=True)
                return True
            except Exception:
                return False

        def _load():
            try:
                t_total = time.perf_counter()
                model_id = DISTIL_MODELS[distil_model_key]["hf_id"]
                self.current_model_id = model_id

                # ── Step 0: HuggingFace login ─────────────────────────────
                self.loading_status = "Authenticating with HuggingFace..."
                self.loading_progress = 3
                huggingface_hub.login(token=hf_token)
                print("HuggingFace login OK")

                # ── Step 1: Distil-Whisper ────────────────────────────────
                if _check_cached(model_id):
                    self.loading_status = f"{distil_model_key} found in cache"
                    self.loading_progress = 20
                    print(f"[cache hit] {model_id}")
                else:
                    self.loading_status = f"Downloading {distil_model_key} (~166 MB, first time only — please wait)..."
                    self.loading_progress = 8
                    print(f"[downloading] {model_id}")
                    # from_pretrained handles Windows-safe caching (no symlink issues)
                    AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float32)
                    self.loading_status = f"{distil_model_key} download complete"
                    self.loading_progress = 20
                    print(f"[done] {model_id}")

                # ── Step 2: Load Distil-Whisper into memory ───────────────
                self.loading_status = f"Loading {distil_model_key} into memory..."
                self.loading_progress = 25
                print("Loading Distil-Whisper weights into memory...")
                t0 = time.perf_counter()

                dtype = torch.float16 if self.device_str == "cuda" else torch.float32
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype)
                model.to(self.device)
                processor = AutoProcessor.from_pretrained(model_id)

                self.asr_pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype,
                    device=self.device,
                    chunk_length_s=30,
                    batch_size=1,
                )
                self.timing['model_load_distil_whisper_s'] = round(time.perf_counter() - t0, 3)
                self.loading_progress = 45
                print(f"Distil-Whisper ready in {self.timing['model_load_distil_whisper_s']}s")

                # ── Step 3: pyannote diarization ──────────────────────────
                self.loading_status = "Loading speaker diarization (pyannote 3.1) — downloading if first run..."
                self.loading_progress = 50
                print("Loading pyannote diarization pipeline...")
                t0 = time.perf_counter()
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                )
                self.diarization_pipeline.to(self.device)
                self.timing['model_load_diarization_s'] = round(time.perf_counter() - t0, 3)
                self.loading_progress = 78
                print(f"Diarization ready in {self.timing['model_load_diarization_s']}s")

                # ── Step 4: VAD ───────────────────────────────────────────
                self.loading_status = "Loading Voice Activity Detection — downloading if first run..."
                self.loading_progress = 82
                print("Loading pyannote VAD...")
                t0 = time.perf_counter()
                vad_model = Model.from_pretrained(
                    "pyannote/segmentation-3.0",
                    use_auth_token=hf_token,
                )
                self.vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
                self.vad_pipeline.instantiate({
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.3
                })
                self.timing['model_load_vad_s'] = round(time.perf_counter() - t0, 3)
                self.timing['model_load_total_s'] = round(time.perf_counter() - t_total, 3)
                self.loading_progress = 95
                print(f"VAD ready in {self.timing['model_load_vad_s']}s")
                print(f"All models loaded in {self.timing['model_load_total_s']}s total")

                self.loading_progress = 100
                self.loading_status = "All models loaded successfully"
                self.models_loaded = True

            except Exception as e:
                self.loading_status = f"Error: {str(e)}"
                self.loading_progress = 0
                self.models_loaded = False
                print(f"Model loading failed: {e}")

        thread = threading.Thread(target=_load)
        thread.daemon = True
        thread.start()
        return True

    def get_loading_status(self):
        return {
            'progress': self.loading_progress,
            'status': self.loading_status,
            'models_loaded': self.models_loaded,
        }

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────

    def preprocess_audio(self, audio_path, target_sr=16000):
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            audio = librosa.util.normalize(audio)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            preprocessed_path = audio_path.replace('.wav', '_preprocessed.wav')
            if not preprocessed_path.endswith('.wav'):
                preprocessed_path += '_preprocessed.wav'
            sf.write(preprocessed_path, audio, target_sr)
            return preprocessed_path, len(audio) / target_sr
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return None, str(e)

    # ── Stage 2: VAD ─────────────────────────────────────────────────────────

    def detect_voice_activity(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_energy = torch.mean(torch.abs(waveform)).item()
            print(f"Audio energy: {audio_energy:.6f}")
            if audio_energy < 0.001:
                print("Audio too quiet — likely silent")
                return [], []

            vad_result = self.vad_pipeline({"waveform": waveform, "sample_rate": sample_rate})
            speech_segments = []
            silence_segments = []
            for speech in vad_result.get_timeline():
                if speech.duration >= 0.3:
                    speech_segments.append({
                        'type': 'speech',
                        'start': round(speech.start, 2),
                        'end': round(speech.end, 2),
                        'duration': round(speech.duration, 2),
                    })
            return speech_segments, silence_segments
        except Exception as e:
            print(f"VAD failed: {e}")
            return [], []

    # ── Stage 3: ASR with Distil-Whisper ────────────────────────────────────

    def transcribe_audio(self, audio_path, language=None):
        """
        Transcribe using Distil-Whisper HuggingFace pipeline.
        Returns a dict shaped like Whisper's output so combine_transcript_and_diarization
        works identically with both backends.
        """
        try:
            # Load audio as numpy array (16 kHz mono)
            audio_np, _ = librosa.load(audio_path, sr=16000)

            generate_kwargs = {"return_timestamps": True}
            if language:
                generate_kwargs["language"] = language

            # Run Distil-Whisper
            result = self.asr_pipe(
                {"raw": audio_np, "sampling_rate": 16000},
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )

            # Convert HuggingFace chunks → Whisper-style segments
            # HuggingFace: result['chunks'] = [{'timestamp': (start, end), 'text': '...'}]
            # Whisper:     result['segments'] = [{'start': ..., 'end': ..., 'text': '...'}]
            chunks = result.get("chunks", [])
            segments = []
            for chunk in chunks:
                ts = chunk.get("timestamp", (None, None))
                start = float(ts[0]) if ts[0] is not None else 0.0
                end   = float(ts[1]) if ts[1] is not None else start + 1.0
                text  = chunk.get("text", "").strip()
                if text:
                    segments.append({
                        "start": round(start, 3),
                        "end":   round(end, 3),
                        "text":  text,
                    })

            return {"text": result.get("text", ""), "segments": segments}

        except Exception as e:
            print(f"Transcription failed: {e}")
            return None

    # ── Stage 4: Diarization ─────────────────────────────────────────────────

    def perform_diarization(self, audio_path, min_speakers=None, max_speakers=None):
        try:
            kwargs = {}
            if min_speakers:
                kwargs['min_speakers'] = min_speakers
            if max_speakers:
                kwargs['max_speakers'] = max_speakers
            return self.diarization_pipeline(audio_path, **kwargs)
        except Exception as e:
            print(f"Diarization failed: {e}")
            return None

    # ── Combine ───────────────────────────────────────────────────────────────

    def combine_transcript_and_diarization(self, transcript_result, diarization):
        try:
            segments = transcript_result.get('segments', [])
            combined = []
            for segment in segments:
                s_start    = segment['start']
                s_end      = segment['end']
                s_duration = s_end - s_start
                s_text     = segment['text'].strip()

                if s_duration < 0.5:
                    continue
                word_count = len(s_text.split())
                if word_count < 2 and s_duration < 2.0:
                    continue

                # Find overlapping speaker in diarization
                speaker = 'Unknown'
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    if (s_start >= turn.start and s_start < turn.end) or \
                       (s_end > turn.start and s_end <= turn.end) or \
                       (s_start <= turn.start and s_end >= turn.end):
                        speaker = spk
                        break

                if s_text:
                    combined.append({
                        'speaker':    speaker,
                        'start':      round(s_start, 2),
                        'end':        round(s_end, 2),
                        'duration':   round(s_duration, 2),
                        'text':       s_text,
                        'start_time': str(timedelta(seconds=int(s_start))),
                        'end_time':   str(timedelta(seconds=int(s_end))),
                        'word_count': word_count,
                    })
            return combined
        except Exception as e:
            print(f"Combine failed: {e}")
            return []

    def create_timeline(self, segments, silence_segments, duration):
        events = []
        for seg in segments:
            events.append({'start': seg['start'], 'end': seg['end'],
                           'type': 'speech', 'speaker': seg['speaker'],
                           'text': seg['text'], 'duration': seg['duration']})
        for sil in silence_segments:
            if sil['duration'] >= 0.5:
                events.append({'start': sil['start'], 'end': sil['end'],
                               'type': 'silence', 'speaker': None,
                               'text': '[SILENCE]', 'duration': sil['duration']})
        events.sort(key=lambda x: x['start'])
        return events

    # ── Full pipeline ────────────────────────────────────────────────────────

    def analyze_audio(self, audio_path, min_speakers=None, max_speakers=None, language=None):
        if not self.models_loaded:
            return {'error': 'Models not loaded'}

        try:
            t_pipeline = time.perf_counter()

            # Stage 1 — Preprocessing
            print("Stage 1: Preprocessing...")
            t0 = time.perf_counter()
            processed_path, duration = self.preprocess_audio(audio_path)
            if processed_path is None:
                return {'error': f'Preprocessing failed: {duration}'}
            self.timing['stage_preprocessing_s'] = round(time.perf_counter() - t0, 3)
            print(f"  Done in {self.timing['stage_preprocessing_s']}s  | Duration: {duration:.2f}s")

            results = {'duration': duration}

            # Stage 2 — VAD
            print("Stage 2: Voice Activity Detection...")
            t0 = time.perf_counter()
            speech_segs, silence_segs = self.detect_voice_activity(processed_path)
            self.timing['stage_vad_s'] = round(time.perf_counter() - t0, 3)
            results['speech_segments']  = speech_segs
            results['silence_segments'] = silence_segs

            total_speech = sum(s['duration'] for s in speech_segs)
            speech_ratio = total_speech / duration if duration > 0 else 0
            print(f"  Done in {self.timing['stage_vad_s']}s | Speech: {total_speech:.1f}s ({speech_ratio*100:.1f}%)")

            if speech_ratio < 0.05 or total_speech < 1.0:
                print("No significant speech found.")
                results.update({'segments': [], 'timeline': [], 'unique_speakers_count': 0,
                                'warning': 'No significant speech detected in this audio file'})
                if os.path.exists(processed_path):
                    os.unlink(processed_path)
                return results

            # Stage 3 — ASR (Distil-Whisper)
            print("Stage 3: Distil-Whisper ASR...")
            t0 = time.perf_counter()
            transcript_result = self.transcribe_audio(processed_path, language=language)
            self.timing['stage_asr_s'] = round(time.perf_counter() - t0, 3)
            if transcript_result is None:
                return {'error': 'Transcription failed'}
            print(f"  Done in {self.timing['stage_asr_s']}s | Segments: {len(transcript_result.get('segments', []))}")

            # Stage 4 — Diarization (pyannote 3.1 — unchanged)
            print("Stage 4: Speaker Diarization (pyannote 3.1)...")
            t0 = time.perf_counter()
            diarization = self.perform_diarization(processed_path, min_speakers, max_speakers)
            self.timing['stage_diarization_s'] = round(time.perf_counter() - t0, 3)
            if diarization is None:
                return {'error': 'Diarization failed'}
            print(f"  Done in {self.timing['stage_diarization_s']}s")

            # Combine
            print("Combining transcript + speaker labels...")
            combined = self.combine_transcript_and_diarization(transcript_result, diarization)
            results['segments'] = combined
            results['timeline'] = self.create_timeline(combined, silence_segs, duration)

            unique_speakers = set(s['speaker'] for s in combined if s['speaker'] != 'Unknown')
            results['unique_speakers_count'] = len(unique_speakers)

            # Timing summary
            self.timing['pipeline_total_s']  = round(time.perf_counter() - t_pipeline, 3)
            self.timing['real_time_factor']  = round(self.timing['pipeline_total_s'] / duration, 3) if duration > 0 else None
            self.timing['audio_duration_s']  = round(duration, 3)
            self.timing['asr_model']         = self.current_model_id

            print("\n========== TIMING REPORT ==========")
            print(f"  ASR model           : {self.current_model_id}")
            print(f"  Audio duration      : {self.timing['audio_duration_s']} s")
            print(f"  Preprocessing       : {self.timing['stage_preprocessing_s']} s")
            print(f"  VAD inference       : {self.timing['stage_vad_s']} s")
            print(f"  ASR (Distil-Whisper): {self.timing['stage_asr_s']} s")
            print(f"  Diarization         : {self.timing['stage_diarization_s']} s")
            print(f"  Pipeline total      : {self.timing['pipeline_total_s']} s")
            print(f"  Real-Time Factor    : {self.timing['real_time_factor']}x")
            print(f"  Load - Distil-Whisper: {self.timing.get('model_load_distil_whisper_s','N/A')} s")
            print(f"  Load - Diarization  : {self.timing.get('model_load_diarization_s','N/A')} s")
            print(f"  Load - VAD          : {self.timing.get('model_load_vad_s','N/A')} s")
            print(f"  Load - Total        : {self.timing.get('model_load_total_s','N/A')} s")
            print("====================================\n")

            results['timing'] = dict(self.timing)
            print(f"Complete: {len(combined)} segments, {results['unique_speakers_count']} speakers")

            if os.path.exists(processed_path):
                os.unlink(processed_path)

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"Analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

analyzer = SpeechAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return jsonify({'error': 'File too large. Max 500 MB'}), 413

@app.errorhandler(500)
def handle_500(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({'error': 'Not found'}), 404


@app.route('/')
def index():
    device_info = {
        'device':       analyzer.device_str.upper(),
        'cuda_available': torch.cuda.is_available(),
        'models_loaded':  analyzer.models_loaded,
        'current_model':  analyzer.current_model_id or 'None',
        'has_env_token':  bool(os.getenv('HF_TOKEN', '').strip()),
    }
    return render_template('index.html', device_info=device_info, distil_models=DISTIL_MODELS)


@app.route('/load_models', methods=['POST', 'OPTIONS'])
def load_models():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        hf_token = os.getenv('HF_TOKEN', '').strip()
        if not hf_token:
            return jsonify({'success': False, 'error': 'HF_TOKEN not found in .env file'}), 400

        data = request.get_json(silent=True) or {}
        distil_model = data.get('distil_model') or request.form.get('distil_model', 'distil-small.en')

        if distil_model not in DISTIL_MODELS:
            return jsonify({'success': False, 'error': f'Unknown model: {distil_model}'}), 400

        print(f"Loading model: {distil_model}")
        analyzer.load_models_async(hf_token, distil_model)
        return jsonify({'success': True, 'message': f'Loading {distil_model}...'}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/loading_status')
def loading_status():
    return jsonify(analyzer.get_loading_status())


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 200
    if not analyzer.models_loaded:
        return jsonify({'error': 'Models not loaded yet'}), 400
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not filename.lower().endswith('.wav'):
            file.save(filepath)
            audio = AudioSegment.from_file(filepath)
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format='wav')
            os.unlink(filepath)
            filepath = wav_path
            filename  = os.path.basename(wav_path)
        else:
            file.save(filepath)

        min_speakers = request.form.get('min_speakers', type=int)
        max_speakers = request.form.get('max_speakers', type=int)
        language     = request.form.get('language') or None

        results = analyzer.analyze_audio(filepath, min_speakers, max_speakers, language)
        if 'error' in results:
            return jsonify({'error': results['error']}), 500

        results['filename'] = filename
        return render_template('results.html', results=results)

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except:
                pass


@app.route('/timing_report')
def timing_report():
    if not analyzer.timing:
        return jsonify({'message': 'No timing data yet — run an analysis first.'}), 200
    return jsonify({
        'platform': {
            'python_version': sys.version,
            'device': analyzer.device_str.upper(),
            'cuda_available': torch.cuda.is_available(),
        },
        'asr_model': analyzer.current_model_id,
        'model_loading_s': {
            'distil_whisper': analyzer.timing.get('model_load_distil_whisper_s'),
            'diarization':    analyzer.timing.get('model_load_diarization_s'),
            'vad':            analyzer.timing.get('model_load_vad_s'),
            'total':          analyzer.timing.get('model_load_total_s'),
        },
        'pipeline_stages_s': {
            'preprocessing':  analyzer.timing.get('stage_preprocessing_s'),
            'vad_inference':  analyzer.timing.get('stage_vad_s'),
            'asr_inference':  analyzer.timing.get('stage_asr_s'),
            'diarization':    analyzer.timing.get('stage_diarization_s'),
            'total':          analyzer.timing.get('pipeline_total_s'),
        },
        'audio_duration_s': analyzer.timing.get('audio_duration_s'),
        'real_time_factor': analyzer.timing.get('real_time_factor'),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001,
            threaded=True, use_reloader=False)
