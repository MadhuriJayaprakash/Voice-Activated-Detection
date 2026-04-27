import sys, types, io

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

for _mod in ['k2', 'speechbrain.integrations.k2_fsa']:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
import os
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

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, '.env'), override=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'moonshine-thesis-2025'
app.config['UPLOAD_FOLDER'] = os.path.join(_HERE, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}})

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

# Moonshine models — English-only, edge-optimised ASR by Useful Sensors
MOONSHINE_MODELS = {
    'tiny': 'usefulsensors/moonshine-tiny',   # ~27M params
    'base': 'usefulsensors/moonshine-base',   # ~61M params
}
DEFAULT_MODEL_SIZE = 'base'
ASR_MODEL_ID  = MOONSHINE_MODELS[DEFAULT_MODEL_SIZE]
ASR_MODEL_TAG = f"Moonshine {DEFAULT_MODEL_SIZE.capitalize()} (Useful Sensors)"


class StageMonitor:
    """Measures RAM, CPU, and estimated energy for one pipeline stage."""
    _TDP_W = 65.0  # Conservative CPU TDP estimate in Watts

    def __init__(self, name):
        self.name          = name
        self.proc          = psutil.Process(os.getpid())
        self.ram_before_mb = 0.0
        self.ram_after_mb  = 0.0
        self.ram_delta_mb  = 0.0
        self.ram_peak_mb   = 0.0
        self.cpu_samples   = []
        self.cpu_avg_pct   = 0.0
        self.duration_s    = 0.0
        self.energy_j      = 0.0
        self.energy_wh     = 0.0
        self._stop_event   = threading.Event()
        self._poll_thread  = None

    def _poll(self):
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
        self._t0           = time.perf_counter()
        psutil.cpu_percent(interval=None)
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll, daemon=True)
        self._poll_thread.start()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        self._poll_thread.join(timeout=1)
        self.duration_s   = time.perf_counter() - self._t0
        self.ram_after_mb = self.proc.memory_info().rss / 1024 / 1024
        self.ram_delta_mb = self.ram_after_mb - self.ram_before_mb
        self.cpu_avg_pct  = (sum(self.cpu_samples) / len(self.cpu_samples)
                             if self.cpu_samples else 0.0)
        power_w        = self._TDP_W * (self.cpu_avg_pct / 100.0)
        self.energy_j  = round(power_w * self.duration_s, 4)
        self.energy_wh = round(self.energy_j / 3600, 6)

    def to_dict(self):
        return {
            'stage':         self.name,
            'duration_s':    round(self.duration_s,    3),
            'ram_before_mb': round(self.ram_before_mb, 1),
            'ram_after_mb':  round(self.ram_after_mb,  1),
            'ram_delta_mb':  round(self.ram_delta_mb,  1),
            'ram_peak_mb':   round(self.ram_peak_mb,   1),
            'cpu_avg_pct':   round(self.cpu_avg_pct,   1),
            'energy_j':      self.energy_j,
            'energy_wh':     self.energy_wh,
        }


class MoonshineAnalyzer:
    def __init__(self):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_pipe             = None
        self.diarization_pipeline = None
        self.vad_pipeline         = None
        self.models_loaded    = False
        self.loading_progress = 0
        self.loading_status   = "Not started"
        self.model_size       = DEFAULT_MODEL_SIZE
        self.asr_model_id     = ASR_MODEL_ID
        self.timing           = {}

    # ── Model loading ─────────────────────────────────────────────────────────
    def load_models_async(self, hf_token, model_size='base'):
        def _load():
            try:
                from transformers import pipeline as hf_pipeline, AutoProcessor

                t_total = time.perf_counter()
                self.model_size   = model_size
                self.asr_model_id = MOONSHINE_MODELS.get(model_size, MOONSHINE_MODELS['base'])

                os.environ['HF_TOKEN'] = hf_token
                os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
                print(f"HF token set: {hf_token[:8]}...")

                # ── Moonshine ASR ─────────────────────────────────────────────
                self.loading_status   = f"Downloading Moonshine {model_size} ({self.asr_model_id})..."
                self.loading_progress = 10
                print(f"Loading {self.asr_model_id}...")
                t0 = time.perf_counter()

                from transformers import MoonshineForConditionalGeneration, AutoProcessor

                dtype = torch.float16 if self.device_str == "cuda" else torch.float32
                model = MoonshineForConditionalGeneration.from_pretrained(
                    self.asr_model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    token=hf_token,
                )
                model.to(self.device)

                processor = AutoProcessor.from_pretrained(
                    self.asr_model_id, token=hf_token
                )

                self.asr_pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype,
                    device=self.device,
                )
                self.timing['model_load_asr_s'] = round(time.perf_counter() - t0, 3)
                self.loading_progress = 45
                print(f"Moonshine ASR ready in {self.timing['model_load_asr_s']}s")

                # ── Speaker Diarization (pyannote 3.1) ────────────────────────
                self.loading_status   = "Loading speaker diarization (PyAnnote 3.1)..."
                self.loading_progress = 60
                print("Loading diarization pipeline...")
                t0 = time.perf_counter()
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                )
                self.diarization_pipeline.to(self.device)
                self.timing['model_load_diarization_s'] = round(time.perf_counter() - t0, 3)
                print(f"Diarization ready in {self.timing['model_load_diarization_s']}s")

                # ── VAD (pyannote segmentation-3.0) ───────────────────────────
                self.loading_status   = "Loading Voice Activity Detection..."
                self.loading_progress = 80
                print("Loading VAD...")
                t0 = time.perf_counter()
                vad_model = Model.from_pretrained(
                    "pyannote/segmentation-3.0",
                    use_auth_token=hf_token,
                )
                self.vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
                self.vad_pipeline.instantiate({
                    "min_duration_on":  0.1,
                    "min_duration_off": 0.3,
                })
                self.timing['model_load_vad_s']   = round(time.perf_counter() - t0, 3)
                self.timing['model_load_total_s'] = round(time.perf_counter() - t_total, 3)
                print(f"VAD ready in {self.timing['model_load_vad_s']}s")
                print(f"Total model load: {self.timing['model_load_total_s']}s")

                self.loading_progress = 100
                self.loading_status   = "All models loaded successfully"
                self.models_loaded    = True
                print("All models ready — waiting for uploads.")

            except Exception as e:
                import traceback; traceback.print_exc()
                self.loading_status   = f"Error: {str(e)}"
                self.loading_progress = 0
                self.models_loaded    = False
                print(f"Model loading failed: {e}")

        threading.Thread(target=_load, daemon=True).start()
        return True

    def get_loading_status(self):
        return {
            'progress':      self.loading_progress,
            'status':        self.loading_status,
            'models_loaded': self.models_loaded,
        }

    # ── Audio helpers ─────────────────────────────────────────────────────────
    def preprocess_audio(self, audio_path, target_sr=16000):
        try:
            audio, _ = librosa.load(audio_path, sr=target_sr)
            audio    = librosa.util.normalize(audio)
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            out_path = audio_path.replace('.wav', '_preprocessed.wav')
            sf.write(out_path, audio, target_sr)
            return out_path, len(audio) / target_sr
        except Exception as e:
            import traceback; traceback.print_exc()
            return None, str(e)

    def detect_voice_activity(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            energy = torch.mean(torch.abs(waveform)).item()
            print(f"Audio energy: {energy:.6f}")
            if energy < 0.001:
                return [], []
            vad_result = self.vad_pipeline({"waveform": waveform, "sample_rate": sr})
            segs = []
            for seg in vad_result.get_timeline():
                if seg.duration >= 0.3:
                    segs.append({
                        'type':     'speech',
                        'start':    round(seg.start,    2),
                        'end':      round(seg.end,      2),
                        'duration': round(seg.duration, 2),
                    })
            return segs, []
        except Exception as e:
            print(f"VAD failed: {e}")
            return [], []

    # ── ASR ───────────────────────────────────────────────────────────────────
    def transcribe_segments(self, audio_path, speech_segments):
        """Transcribe each VAD segment with Moonshine (English only)."""
        results = []
        audio, sr = librosa.load(audio_path, sr=16000)

        for seg in speech_segments:
            chunk = audio[int(seg['start'] * sr): int(seg['end'] * sr)]
            if len(chunk) < int(sr * 0.3):
                continue

            tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, chunk, sr)
                    tmp = f.name

                out  = self.asr_pipe(tmp)
                text = out.get("text", "").strip()
                if text:
                    results.append({
                        'start':    seg['start'],
                        'end':      seg['end'],
                        'duration': seg['duration'],
                        'text':     text,
                    })
            except Exception as e:
                print(f"ASR failed [{seg['start']:.1f}-{seg['end']:.1f}s]: {e}")
            finally:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)

        return results

    # ── Diarization ───────────────────────────────────────────────────────────
    def perform_diarization(self, audio_path, min_speakers=None, max_speakers=None):
        try:
            kwargs = {}
            if min_speakers: kwargs['min_speakers'] = min_speakers
            if max_speakers: kwargs['max_speakers'] = max_speakers
            return self.diarization_pipeline(audio_path, **kwargs)
        except Exception as e:
            print(f"Diarization failed: {e}")
            return None

    def combine_transcripts_and_diarization(self, transcript_segments, diarization):
        combined = []
        for seg in transcript_segments:
            speaker      = 'Unknown'
            best_overlap = 0.0
            if diarization:
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    overlap = max(0.0,
                        min(seg['end'], turn.end) - max(seg['start'], turn.start))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        speaker      = spk
            combined.append({
                'speaker':    speaker,
                'start':      round(seg['start'],    2),
                'end':        round(seg['end'],      2),
                'duration':   round(seg['duration'], 2),
                'text':       seg['text'],
                'start_time': str(timedelta(seconds=int(seg['start']))),
                'end_time':   str(timedelta(seconds=int(seg['end']))),
                'word_count': len(seg['text'].split()),
            })
        return combined

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

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def analyze_audio(self, audio_path, min_speakers=None, max_speakers=None):
        if not self.models_loaded:
            return {'error': 'Models not loaded'}
        try:
            t_pipe        = time.perf_counter()
            resource_stats = {}

            # Stage 1 — Preprocess
            print("\n[Stage 1] Preprocessing...")
            with StageMonitor("Preprocessing") as m1:
                proc_path, duration = self.preprocess_audio(audio_path)
            if proc_path is None:
                return {'error': f'Preprocessing failed: {duration}'}
            self.timing['stage_preprocessing_s'] = round(m1.duration_s, 3)
            resource_stats['preprocessing'] = m1.to_dict()
            print(f"  {m1.duration_s:.3f}s | audio={duration:.2f}s | "
                  f"RAM {m1.ram_delta_mb:+.1f} MB | CPU {m1.cpu_avg_pct:.1f}% | "
                  f"Energy {m1.energy_j:.3f} J")

            results = {'duration': duration}

            # Stage 2 — VAD
            print("[Stage 2] Voice Activity Detection...")
            with StageMonitor("VAD") as m2:
                speech_segs, silence_segs = self.detect_voice_activity(proc_path)
            self.timing['stage_vad_s']  = round(m2.duration_s, 3)
            resource_stats['vad']       = m2.to_dict()
            results['speech_segments']  = speech_segs
            results['silence_segments'] = silence_segs

            total_speech = sum(s['duration'] for s in speech_segs)
            ratio = total_speech / duration if duration > 0 else 0
            print(f"  {m2.duration_s:.3f}s | speech={total_speech:.2f}s ({ratio*100:.1f}%) | "
                  f"RAM {m2.ram_delta_mb:+.1f} MB | CPU {m2.cpu_avg_pct:.1f}% | "
                  f"Energy {m2.energy_j:.3f} J")

            if ratio < 0.05 or total_speech < 1.0:
                results.update({'segments': [], 'timeline': [],
                                'unique_speakers_count': 0,
                                'warning': 'No significant speech detected'})
                if os.path.exists(proc_path): os.unlink(proc_path)
                return results

            # Stage 3 — ASR
            print(f"[Stage 3] Moonshine ASR ({len(speech_segs)} segments)...")
            with StageMonitor("ASR (Moonshine)") as m3:
                transcript_segs = self.transcribe_segments(proc_path, speech_segs)
            self.timing['stage_asr_s'] = round(m3.duration_s, 3)
            resource_stats['asr']      = m3.to_dict()
            print(f"  {m3.duration_s:.3f}s | {len(transcript_segs)} segments | "
                  f"RAM {m3.ram_delta_mb:+.1f} MB | CPU {m3.cpu_avg_pct:.1f}% | "
                  f"Energy {m3.energy_j:.3f} J")

            # Stage 4 — Diarization
            print("[Stage 4] Speaker Diarization...")
            with StageMonitor("Diarization") as m4:
                diarization = self.perform_diarization(proc_path, min_speakers, max_speakers)
            self.timing['stage_diarization_s'] = round(m4.duration_s, 3)
            resource_stats['diarization']      = m4.to_dict()
            print(f"  {m4.duration_s:.3f}s | "
                  f"RAM {m4.ram_delta_mb:+.1f} MB | CPU {m4.cpu_avg_pct:.1f}% | "
                  f"Energy {m4.energy_j:.3f} J")

            # Aggregate totals
            total_energy_j  = sum(resource_stats[s]['energy_j']  for s in resource_stats)
            total_energy_wh = sum(resource_stats[s]['energy_wh'] for s in resource_stats)
            peak_ram_mb     = max(resource_stats[s]['ram_peak_mb'] for s in resource_stats)
            avg_cpu_pct     = (sum(resource_stats[s]['cpu_avg_pct'] for s in resource_stats)
                               / len(resource_stats))
            resource_stats['pipeline_total'] = {
                'total_energy_j':  round(total_energy_j,  4),
                'total_energy_wh': round(total_energy_wh, 6),
                'peak_ram_mb':     round(peak_ram_mb,     1),
                'avg_cpu_pct':     round(avg_cpu_pct,     1),
            }

            # Combine
            combined = self.combine_transcripts_and_diarization(transcript_segs, diarization)
            results['segments']              = combined
            results['timeline']              = self.create_timeline(combined, silence_segs, duration)
            results['unique_speakers_count'] = len(
                {s['speaker'] for s in combined if s['speaker'] != 'Unknown'})
            results['model']          = self.asr_model_id
            results['model_size']     = self.model_size
            results['language']       = 'en'
            results['resource_stats'] = resource_stats

            self.timing['pipeline_total_s'] = round(time.perf_counter() - t_pipe, 3)
            self.timing['real_time_factor'] = (
                round(self.timing['pipeline_total_s'] / duration, 3) if duration > 0 else None)
            self.timing['audio_duration_s']  = round(duration, 3)
            self.timing['resource_stats']    = resource_stats

            pt = resource_stats['pipeline_total']
            print("\n========== TIMING + RESOURCE REPORT ==========")
            print(f"  ASR Model        : {self.asr_model_id}")
            print(f"  Audio duration   : {self.timing['audio_duration_s']} s")
            print(f"  {'Stage':<22} {'Time(s)':>8} {'RAM d(MB)':>10} {'CPU%':>7} {'Energy(J)':>10}")
            print(f"  {'-'*60}")
            for key, label in [('preprocessing','Preprocessing'), ('vad','VAD'),
                                ('asr','ASR (Moonshine)'),  ('diarization','Diarization')]:
                s = resource_stats[key]
                print(f"  {label:<22} {s['duration_s']:>8.3f} {s['ram_delta_mb']:>+10.1f} "
                      f"{s['cpu_avg_pct']:>7.1f} {s['energy_j']:>10.3f}")
            print(f"  {'-'*60}")
            print(f"  {'TOTAL':<22} {self.timing['pipeline_total_s']:>8.3f} "
                  f"{'peak:'+str(pt['peak_ram_mb'])+'MB':>10} {pt['avg_cpu_pct']:>7.1f} "
                  f"{pt['total_energy_j']:>10.3f}")
            print(f"  RTF              : {self.timing['real_time_factor']}x")
            print(f"  Total Energy     : {pt['total_energy_j']} J ({pt['total_energy_wh']} Wh)")
            print("===============================================\n")

            results['timing'] = dict(self.timing)

            if os.path.exists(proc_path): os.unlink(proc_path)
            gc.collect()
            if self.device_str == "cuda": torch.cuda.empty_cache()

            return results

        except Exception as e:
            import traceback; traceback.print_exc()
            return {'error': f'Analysis failed: {str(e)}'}


# ── Global instance ───────────────────────────────────────────────────────────
analyzer = MoonshineAnalyzer()


def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(RequestEntityTooLarge)
def too_large(e):  return jsonify({'error': 'File too large. Max 500 MB'}), 413
@app.errorhandler(500)
def server_err(e): return jsonify({'error': 'Internal server error'}), 500
@app.errorhandler(404)
def not_found(e):  return jsonify({'error': 'Not found'}), 404


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
        device_info={
            'device':        analyzer.device_str.upper(),
            'cuda_available': torch.cuda.is_available(),
            'models_loaded': analyzer.models_loaded,
            'has_env_token': bool(os.getenv('HF_TOKEN', '').strip()),
            'asr_model':     analyzer.asr_model_id,
            'model_size':    analyzer.model_size,
        },
        moonshine_models=MOONSHINE_MODELS,
    )


@app.route('/load_models', methods=['POST', 'OPTIONS'])
def load_models():
    if request.method == 'OPTIONS': return '', 200
    try:
        hf_token = os.getenv('HF_TOKEN', '').strip()
        if not hf_token:
            return jsonify({'success': False, 'error': 'HF_TOKEN not found in .env'}), 400
        data       = request.json if request.is_json else request.form
        model_size = data.get('model_size', 'base')
        print(f"Loading Moonshine {model_size}")
        analyzer.load_models_async(hf_token, model_size)
        return jsonify({'success': True, 'message': 'Loading started in background'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/loading_status')
def loading_status():
    return jsonify(analyzer.get_loading_status())


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS': return '', 200
    if not analyzer.models_loaded:
        return jsonify({'error': 'Models not loaded yet'}), 400
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not filename.lower().endswith('.wav'):
            file.save(filepath)
            AudioSegment.from_file(filepath).export(
                filepath.rsplit('.', 1)[0] + '.wav', format='wav')
            os.unlink(filepath)
            filepath = filepath.rsplit('.', 1)[0] + '.wav'
        else:
            file.save(filepath)

        results = analyzer.analyze_audio(
            filepath,
            min_speakers=request.form.get('min_speakers', type=int),
            max_speakers=request.form.get('max_speakers', type=int),
        )
        if 'error' in results:
            return jsonify({'error': results['error']}), 500

        results['filename'] = filename
        return render_template('results.html', results=results)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try: os.unlink(filepath)
            except: pass


@app.route('/timing_report')
def timing_report():
    if not analyzer.timing:
        return jsonify({'message': 'No timing data yet. Run an analysis first.'}), 200
    return jsonify({
        'platform': {
            'python_version': sys.version,
            'device':         analyzer.device_str.upper(),
            'cuda_available': torch.cuda.is_available(),
            'asr_model':      analyzer.asr_model_id,
            'model_size':     analyzer.model_size,
            'language':       'en (English only)',
        },
        'model_loading_s': {
            'asr':         analyzer.timing.get('model_load_asr_s'),
            'diarization': analyzer.timing.get('model_load_diarization_s'),
            'vad':         analyzer.timing.get('model_load_vad_s'),
            'total':       analyzer.timing.get('model_load_total_s'),
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
    })


# ── Entry point ───────────────────────────────────────────────────────────────
def _open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5002')
    print("   [OK] Browser opened  -->  http://localhost:5002")


def _auto_load():
    time.sleep(3)
    token = os.getenv('HF_TOKEN', '').strip()
    if token and not analyzer.models_loaded:
        print("HF_TOKEN found — auto-loading Moonshine base...")
        analyzer.load_models_async(token, 'base')


if __name__ == '__main__':
    PORT = 5002
    print()
    print("=" * 60)
    print("   MOONSHINE ASR  |  VAD  |  Speaker Diarization")
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
