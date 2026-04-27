import sys, types, io
# Fix Windows console encoding so Unicode chars don't crash print()
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
import webbrowser
import psutil
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

# ── Resource Monitor ──────────────────────────────────────────────────────────
class StageMonitor:
    """
    Context manager that measures RAM, CPU, and estimated energy for one pipeline stage.

    Usage:
        with StageMonitor("ASR") as m:
            run_asr(...)
        print(m.report())
    """
    # Rough CPU TDP estimate (Watts). Reads from psutil if available, else 65 W default.
    _TDP_W = None

    @classmethod
    def _get_tdp(cls):
        if cls._TDP_W is not None:
            return cls._TDP_W
        try:
            freq = psutil.cpu_freq()
            # Very rough estimate: desktop CPUs ~65 W at max freq
            cls._TDP_W = 65.0
        except Exception:
            cls._TDP_W = 65.0
        return cls._TDP_W

    def __init__(self, name):
        self.name = name
        self.proc = psutil.Process(os.getpid())
        # Results (filled after __exit__)
        self.ram_before_mb   = 0.0
        self.ram_after_mb    = 0.0
        self.ram_delta_mb    = 0.0
        self.ram_peak_mb     = 0.0
        self.cpu_samples     = []
        self.cpu_avg_pct     = 0.0
        self.duration_s      = 0.0
        self.energy_j        = 0.0      # Joules
        self.energy_wh       = 0.0      # Watt-hours
        self._stop_event     = threading.Event()
        self._poll_thread    = None

    def _poll(self):
        """Background thread: sample CPU and RAM every 200 ms."""
        peak = self.ram_before_mb
        while not self._stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = self.proc.memory_info().rss / 1024 / 1024
                self.cpu_samples.append(cpu)
                if ram > peak:
                    peak = ram
                self.ram_peak_mb = peak
            except Exception:
                pass
            time.sleep(0.2)

    def __enter__(self):
        self.ram_before_mb = self.proc.memory_info().rss / 1024 / 1024
        self.ram_peak_mb   = self.ram_before_mb
        self._t0 = time.perf_counter()
        psutil.cpu_percent(interval=None)          # prime the counter
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll, daemon=True)
        self._poll_thread.start()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        self._poll_thread.join(timeout=1)
        self.duration_s    = time.perf_counter() - self._t0
        self.ram_after_mb  = self.proc.memory_info().rss / 1024 / 1024
        self.ram_delta_mb  = self.ram_after_mb - self.ram_before_mb
        self.cpu_avg_pct   = (sum(self.cpu_samples) / len(self.cpu_samples)
                              if self.cpu_samples else 0.0)
        # Energy estimate: E = P × t, P = TDP × (cpu% / 100)
        power_w            = self._get_tdp() * (self.cpu_avg_pct / 100.0)
        self.energy_j      = round(power_w * self.duration_s, 4)
        self.energy_wh     = round(self.energy_j / 3600, 6)

    def to_dict(self):
        return {
            "stage":          self.name,
            "duration_s":     round(self.duration_s, 3),
            "ram_before_mb":  round(self.ram_before_mb, 1),
            "ram_after_mb":   round(self.ram_after_mb, 1),
            "ram_delta_mb":   round(self.ram_delta_mb, 1),
            "ram_peak_mb":    round(self.ram_peak_mb, 1),
            "cpu_avg_pct":    round(self.cpu_avg_pct, 1),
            "energy_j":       self.energy_j,
            "energy_wh":      self.energy_wh,
        }

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, '.env'), override=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'distil-whisper-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(_HERE, 'uploads')
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

                # ── Step 0: HuggingFace token setup ──────────────────────
                self.loading_status = "Authenticating with HuggingFace..."
                self.loading_progress = 3
                os.environ['HF_TOKEN'] = hf_token
                os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
                print(f"HF token set: {hf_token[:8]}...")

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
            resource_stats = {}   # per-stage resource data

            # Stage 1 — Preprocessing
            print("Stage 1: Preprocessing...")
            with StageMonitor("Preprocessing") as m1:
                processed_path, duration = self.preprocess_audio(audio_path)
            if processed_path is None:
                return {'error': f'Preprocessing failed: {duration}'}
            self.timing['stage_preprocessing_s'] = round(m1.duration_s, 3)
            resource_stats['preprocessing'] = m1.to_dict()
            print(f"  Done in {m1.duration_s:.3f}s | Duration: {duration:.2f}s | "
                  f"RAM delta: {m1.ram_delta_mb:+.1f} MB | CPU: {m1.cpu_avg_pct:.1f}% | "
                  f"Energy: {m1.energy_j:.3f} J")

            results = {'duration': duration}

            # Stage 2 — VAD
            print("Stage 2: Voice Activity Detection...")
            with StageMonitor("VAD") as m2:
                speech_segs, silence_segs = self.detect_voice_activity(processed_path)
            self.timing['stage_vad_s'] = round(m2.duration_s, 3)
            resource_stats['vad'] = m2.to_dict()
            results['speech_segments']  = speech_segs
            results['silence_segments'] = silence_segs

            total_speech = sum(s['duration'] for s in speech_segs)
            speech_ratio = total_speech / duration if duration > 0 else 0
            print(f"  Done in {m2.duration_s:.3f}s | Speech: {total_speech:.1f}s ({speech_ratio*100:.1f}%) | "
                  f"RAM delta: {m2.ram_delta_mb:+.1f} MB | CPU: {m2.cpu_avg_pct:.1f}% | "
                  f"Energy: {m2.energy_j:.3f} J")

            if speech_ratio < 0.05 or total_speech < 1.0:
                print("No significant speech found.")
                results.update({'segments': [], 'timeline': [], 'unique_speakers_count': 0,
                                'warning': 'No significant speech detected in this audio file'})
                if os.path.exists(processed_path):
                    os.unlink(processed_path)
                return results

            # Stage 3 — ASR (Distil-Whisper)
            print("Stage 3: Distil-Whisper ASR...")
            with StageMonitor("ASR (Distil-Whisper)") as m3:
                transcript_result = self.transcribe_audio(processed_path, language=language)
            self.timing['stage_asr_s'] = round(m3.duration_s, 3)
            resource_stats['asr'] = m3.to_dict()
            if transcript_result is None:
                return {'error': 'Transcription failed'}
            print(f"  Done in {m3.duration_s:.3f}s | Segments: {len(transcript_result.get('segments', []))} | "
                  f"RAM delta: {m3.ram_delta_mb:+.1f} MB | CPU: {m3.cpu_avg_pct:.1f}% | "
                  f"Energy: {m3.energy_j:.3f} J")

            # Stage 4 — Diarization
            print("Stage 4: Speaker Diarization (pyannote 3.1)...")
            with StageMonitor("Diarization") as m4:
                diarization = self.perform_diarization(processed_path, min_speakers, max_speakers)
            self.timing['stage_diarization_s'] = round(m4.duration_s, 3)
            resource_stats['diarization'] = m4.to_dict()
            if diarization is None:
                return {'error': 'Diarization failed'}
            print(f"  Done in {m4.duration_s:.3f}s | "
                  f"RAM delta: {m4.ram_delta_mb:+.1f} MB | CPU: {m4.cpu_avg_pct:.1f}% | "
                  f"Energy: {m4.energy_j:.3f} J")

            # Combine
            print("Combining transcript + speaker labels...")
            combined = self.combine_transcript_and_diarization(transcript_result, diarization)
            results['segments'] = combined
            results['timeline'] = self.create_timeline(combined, silence_segs, duration)

            unique_speakers = set(s['speaker'] for s in combined if s['speaker'] != 'Unknown')
            results['unique_speakers_count'] = len(unique_speakers)

            # ── Aggregate resource totals ─────────────────────────────────
            total_energy_j  = sum(resource_stats[s]['energy_j']  for s in resource_stats)
            total_energy_wh = sum(resource_stats[s]['energy_wh'] for s in resource_stats)
            peak_ram_mb     = max(resource_stats[s]['ram_peak_mb'] for s in resource_stats)
            avg_cpu_pct     = (sum(resource_stats[s]['cpu_avg_pct'] for s in resource_stats)
                               / len(resource_stats))

            resource_stats['pipeline_total'] = {
                'total_energy_j':  round(total_energy_j, 4),
                'total_energy_wh': round(total_energy_wh, 6),
                'peak_ram_mb':     round(peak_ram_mb, 1),
                'avg_cpu_pct':     round(avg_cpu_pct, 1),
            }

            # ── Timing summary ────────────────────────────────────────────
            self.timing['pipeline_total_s']  = round(time.perf_counter() - t_pipeline, 3)
            self.timing['real_time_factor']  = round(self.timing['pipeline_total_s'] / duration, 3) if duration > 0 else None
            self.timing['audio_duration_s']  = round(duration, 3)
            self.timing['asr_model']         = self.current_model_id

            print("\n========== TIMING + RESOURCE REPORT ==========")
            print(f"  ASR model           : {self.current_model_id}")
            print(f"  Audio duration      : {self.timing['audio_duration_s']} s")
            print(f"  {'Stage':<22} {'Time(s)':>8} {'RAM d(MB)':>10} {'CPU%':>7} {'Energy(J)':>10}")
            print(f"  {'-'*60}")
            for key, label in [('preprocessing','Preprocessing'),('vad','VAD'),
                                ('asr','ASR (Distil-Whisper)'),('diarization','Diarization')]:
                s = resource_stats[key]
                print(f"  {label:<22} {s['duration_s']:>8.3f} {s['ram_delta_mb']:>+10.1f} "
                      f"{s['cpu_avg_pct']:>7.1f} {s['energy_j']:>10.3f}")
            print(f"  {'-'*60}")
            pt = resource_stats['pipeline_total']
            print(f"  {'TOTAL':<22} {self.timing['pipeline_total_s']:>8.3f} "
                  f"{'peak:'+str(pt['peak_ram_mb'])+'MB':>10} {pt['avg_cpu_pct']:>7.1f} "
                  f"{pt['total_energy_j']:>10.3f}")
            print(f"  Real-Time Factor    : {self.timing['real_time_factor']}x")
            print(f"  Total Energy        : {pt['total_energy_j']} J  "
                  f"({pt['total_energy_wh']} Wh)")
            print("================================================\n")

            results['timing']         = dict(self.timing)
            results['resource_stats'] = resource_stats
            self.timing['resource_stats'] = resource_stats   # also store for /timing_report
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

    proc = psutil.Process(os.getpid())
    sys_ram_total = round(psutil.virtual_memory().total / 1024 / 1024, 1)
    sys_ram_used  = round(psutil.virtual_memory().used  / 1024 / 1024, 1)
    proc_ram_mb   = round(proc.memory_info().rss / 1024 / 1024, 1)

    return jsonify({
        'platform': {
            'python_version':  sys.version,
            'device':          analyzer.device_str.upper(),
            'cuda_available':  torch.cuda.is_available(),
            'cpu_count':       psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'system_ram_total_mb': sys_ram_total,
            'system_ram_used_mb':  sys_ram_used,
            'process_ram_mb':      proc_ram_mb,
        },
        'asr_model': analyzer.current_model_id,
        'model_loading_s': {
            'distil_whisper': analyzer.timing.get('model_load_distil_whisper_s'),
            'diarization':    analyzer.timing.get('model_load_diarization_s'),
            'vad':            analyzer.timing.get('model_load_vad_s'),
            'total':          analyzer.timing.get('model_load_total_s'),
        },
        'pipeline_stages_s': {
            'preprocessing': analyzer.timing.get('stage_preprocessing_s'),
            'vad_inference': analyzer.timing.get('stage_vad_s'),
            'asr_inference': analyzer.timing.get('stage_asr_s'),
            'diarization':   analyzer.timing.get('stage_diarization_s'),
            'total':         analyzer.timing.get('pipeline_total_s'),
        },
        'audio_duration_s': analyzer.timing.get('audio_duration_s'),
        'real_time_factor': analyzer.timing.get('real_time_factor'),
        'resource_stats':   analyzer.timing.get('resource_stats', {}),
    })


def _open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5003')
    print("   [OK] Browser opened  -->  http://localhost:5003")


def _auto_load():
    time.sleep(3)
    token = os.getenv('HF_TOKEN', '').strip()
    if token and not analyzer.models_loaded:
        print("   [INFO] HF_TOKEN found — auto-loading distil-small.en...")
        analyzer.load_models_async(token, 'distil-small.en')


if __name__ == '__main__':
    PORT = 5003
    print()
    print("=" * 60)
    print("   DISTIL-WHISPER  |  VAD  |  Speaker Diarization")
    print("=" * 60)
    print()
    print(f"   App running at  -->  http://localhost:{PORT}")
    print()
    print("   [INFO] Browser will open automatically in 2 seconds")
    print("   [INFO] Models will start loading automatically")
    print("   [INFO] Press Ctrl+C to stop the server")
    print()
    print("=" * 60)
    print()

    threading.Thread(target=_open_browser, daemon=True).start()
    threading.Thread(target=_auto_load,    daemon=True).start()

    app.run(debug=False, host='0.0.0.0', port=PORT,
            threaded=True, use_reloader=False)
