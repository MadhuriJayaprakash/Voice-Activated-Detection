import sys, types

# Stub out k2_fsa before speechbrain lazy-loader triggers it and crashes
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
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
# Load .env relative to this file's directory so it works from any working dir
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, '.env'), override=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'canary-qwen-thesis-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

# ASR model used — whisper-large-v3 is the best Windows-compatible open-source
# model on HuggingFace Open ASR Leaderboard (Canary-Qwen requires Linux/NeMo)
ASR_MODEL_ID  = "openai/whisper-large-v3"
ASR_MODEL_TAG = "Whisper Large-v3 (HuggingFace)"

SUPPORTED_LANGS = {
    'en': 'English', 'de': 'German', 'es': 'Spanish', 'fr': 'French',
    'it': 'Italian', 'ja': 'Japanese', 'pt': 'Portuguese', 'zh': 'Chinese',
    'ar': 'Arabic',  'hi': 'Hindi',   'ko': 'Korean',   'ru': 'Russian',
    'nl': 'Dutch',   'pl': 'Polish',  'tr': 'Turkish'
}


class CanaryAnalyzer:
    def __init__(self):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_pipe   = None          # HuggingFace ASR pipeline
        self.diarization_pipeline = None
        self.vad_pipeline         = None
        self.models_loaded    = False
        self.loading_progress = 0
        self.loading_status   = "Not started"
        self.source_lang      = 'en'
        self.target_lang      = 'en'
        self.timing           = {}

    # ── Model loading ─────────────────────────────────────────────────────────
    def load_models_async(self, hf_token, source_lang='en', target_lang='en'):
        """Load all models in a background thread."""
        def _load():
            try:
                from transformers import pipeline as hf_pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

                t_total          = time.perf_counter()
                self.source_lang = source_lang
                self.target_lang = target_lang

                # Set token directly in environment so all HF libraries pick it up
                os.environ['HF_TOKEN'] = hf_token
                os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
                print(f"HF token set: {hf_token[:8]}...")

                # ── Whisper large-v3 via HuggingFace transformers ─────────────
                self.loading_status   = f"Downloading {ASR_MODEL_ID}..."
                self.loading_progress = 10
                print(f"Loading {ASR_MODEL_ID}...")
                t0 = time.perf_counter()

                dtype = torch.float16 if self.device_str == "cuda" else torch.float32
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    ASR_MODEL_ID,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    token=hf_token,
                )
                model.to(self.device)

                processor = AutoProcessor.from_pretrained(ASR_MODEL_ID, token=hf_token)

                self.asr_pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype,
                    device=self.device,
                    return_timestamps=True,
                    chunk_length_s=30,
                    stride_length_s=5,
                )
                self.timing['model_load_asr_s'] = round(time.perf_counter() - t0, 3)
                self.loading_progress = 45
                print(f"ASR model ready in {self.timing['model_load_asr_s']}s")

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
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.3
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
                import traceback
                traceback.print_exc()
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
            'models_loaded': self.models_loaded
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
                        'start':    round(seg.start, 2),
                        'end':      round(seg.end,   2),
                        'duration': round(seg.duration, 2)
                    })
            return segs, []
        except Exception as e:
            print(f"VAD failed: {e}")
            return [], []

    # ── ASR ───────────────────────────────────────────────────────────────────
    def transcribe_segments(self, audio_path, speech_segments):
        """
        Transcribe each VAD speech segment with Whisper large-v3.
        Uses chunk_length_s so long segments are handled gracefully.
        """
        results = []
        audio, sr = librosa.load(audio_path, sr=16000)
        lang = self.source_lang if self.source_lang != self.target_lang else self.source_lang
        task = "translate" if self.source_lang != self.target_lang else "transcribe"

        for seg in speech_segments:
            chunk = audio[int(seg['start'] * sr): int(seg['end'] * sr)]
            if len(chunk) < int(sr * 0.3):
                continue

            tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, chunk, sr)
                    tmp = f.name

                out  = self.asr_pipe(
                    tmp,
                    generate_kwargs={
                        "language": lang,
                        "task":     task,
                    }
                )
                text = out.get("text", "").strip()
                if text:
                    results.append({
                        'start':    seg['start'],
                        'end':      seg['end'],
                        'duration': seg['duration'],
                        'text':     text
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
                'word_count': len(seg['text'].split())
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
            t_pipe = time.perf_counter()

            # Stage 1 — Preprocess
            print("\n[Stage 1] Preprocessing...")
            t0 = time.perf_counter()
            proc_path, duration = self.preprocess_audio(audio_path)
            if proc_path is None:
                return {'error': f'Preprocessing failed: {duration}'}
            self.timing['stage_preprocessing_s'] = round(time.perf_counter() - t0, 3)
            print(f"  {self.timing['stage_preprocessing_s']}s | audio={duration:.2f}s")

            results = {'duration': duration}

            # Stage 2 — VAD
            print("[Stage 2] Voice Activity Detection...")
            t0 = time.perf_counter()
            speech_segs, silence_segs = self.detect_voice_activity(proc_path)
            self.timing['stage_vad_s'] = round(time.perf_counter() - t0, 3)
            results['speech_segments']  = speech_segs
            results['silence_segments'] = silence_segs

            total_speech = sum(s['duration'] for s in speech_segs)
            ratio = total_speech / duration if duration > 0 else 0
            print(f"  {self.timing['stage_vad_s']}s | speech={total_speech:.2f}s ({ratio*100:.1f}%)")

            if ratio < 0.05 or total_speech < 1.0:
                results.update({'segments': [], 'timeline': [],
                                'unique_speakers_count': 0,
                                'warning': 'No significant speech detected'})
                if os.path.exists(proc_path): os.unlink(proc_path)
                return results

            # Stage 3 — ASR
            print(f"[Stage 3] ASR with {ASR_MODEL_TAG} ({len(speech_segs)} segments)...")
            t0 = time.perf_counter()
            transcript_segs = self.transcribe_segments(proc_path, speech_segs)
            self.timing['stage_asr_s'] = round(time.perf_counter() - t0, 3)
            print(f"  {self.timing['stage_asr_s']}s | {len(transcript_segs)} segments transcribed")

            # Stage 4 — Diarization
            print("[Stage 4] Speaker Diarization...")
            t0 = time.perf_counter()
            diarization = self.perform_diarization(proc_path, min_speakers, max_speakers)
            self.timing['stage_diarization_s'] = round(time.perf_counter() - t0, 3)
            print(f"  {self.timing['stage_diarization_s']}s")

            # Combine
            combined = self.combine_transcripts_and_diarization(transcript_segs, diarization)
            results['segments'] = combined
            results['timeline'] = self.create_timeline(combined, silence_segs, duration)
            results['unique_speakers_count'] = len(
                {s['speaker'] for s in combined if s['speaker'] != 'Unknown'})
            results['model']       = ASR_MODEL_ID
            results['source_lang'] = self.source_lang
            results['target_lang'] = self.target_lang

            self.timing['pipeline_total_s']  = round(time.perf_counter() - t_pipe, 3)
            self.timing['real_time_factor']  = (
                round(self.timing['pipeline_total_s'] / duration, 3) if duration > 0 else None)
            self.timing['audio_duration_s']  = round(duration, 3)

            print("\n========== TIMING REPORT ==========")
            print(f"  ASR Model           : {ASR_MODEL_ID}")
            print(f"  Audio duration      : {self.timing['audio_duration_s']} s")
            print(f"  Preprocessing       : {self.timing['stage_preprocessing_s']} s")
            print(f"  VAD inference       : {self.timing['stage_vad_s']} s")
            print(f"  ASR inference       : {self.timing['stage_asr_s']} s")
            print(f"  Diarization         : {self.timing['stage_diarization_s']} s")
            print(f"  Pipeline total      : {self.timing['pipeline_total_s']} s")
            print(f"  Real-Time Factor    : {self.timing['real_time_factor']}x")
            print(f"  Model load - ASR    : {self.timing.get('model_load_asr_s','N/A')} s")
            print(f"  Model load - Diar.  : {self.timing.get('model_load_diarization_s','N/A')} s")
            print(f"  Model load - VAD    : {self.timing.get('model_load_vad_s','N/A')} s")
            print(f"  Model load - Total  : {self.timing.get('model_load_total_s','N/A')} s")
            print("====================================\n")

            results['timing'] = dict(self.timing)

            if os.path.exists(proc_path): os.unlink(proc_path)
            gc.collect()
            if self.device_str == "cuda": torch.cuda.empty_cache()

            return results

        except Exception as e:
            import traceback; traceback.print_exc()
            return {'error': f'Analysis failed: {str(e)}'}


# ── Global instance ───────────────────────────────────────────────────────────
analyzer = CanaryAnalyzer()


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
            'source_lang':   analyzer.source_lang,
            'target_lang':   analyzer.target_lang,
            'asr_model':     ASR_MODEL_ID,
        },
        supported_langs=SUPPORTED_LANGS
    )


@app.route('/load_models', methods=['POST', 'OPTIONS'])
def load_models():
    if request.method == 'OPTIONS': return '', 200
    try:
        hf_token = os.getenv('HF_TOKEN', '').strip()
        if not hf_token:
            return jsonify({'success': False, 'error': 'HF_TOKEN not found in .env'}), 400
        data        = request.json if request.is_json else request.form
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'en')
        print(f"Loading models | src={source_lang} tgt={target_lang}")
        analyzer.load_models_async(hf_token, source_lang, target_lang)
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
        return render_template('results.html', results=results,
                               supported_langs=SUPPORTED_LANGS)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try: os.unlink(filepath)
            except: pass


@app.route('/debug_token')
def debug_token():
    token = os.getenv('HF_TOKEN', '')
    return jsonify({
        'token_first8': token[:8] if token else 'EMPTY',
        'token_len': len(token),
        'env_file': os.path.join(_HERE, '.env'),
        'env_file_exists': os.path.exists(os.path.join(_HERE, '.env')),
    })

@app.route('/timing_report')
def timing_report():
    if not analyzer.timing:
        return jsonify({'message': 'No timing data yet. Run an analysis first.'}), 200
    return jsonify({
        'platform': {
            'python_version':  sys.version,
            'device':          analyzer.device_str.upper(),
            'cuda_available':  torch.cuda.is_available(),
            'asr_model':       ASR_MODEL_ID,
            'source_lang':     analyzer.source_lang,
            'target_lang':     analyzer.target_lang,
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
    webbrowser.open('http://localhost:5001')
    print("\n  Browser opened at http://localhost:5001\n")


def _auto_load():
    """Start model loading automatically when HF_TOKEN is in .env."""
    time.sleep(3)
    token = os.getenv('HF_TOKEN', '').strip()
    if token and not analyzer.models_loaded:
        print("HF_TOKEN found — auto-loading models...")
        analyzer.load_models_async(token, 'en', 'en')


if __name__ == '__main__':
    PORT = 5001
    print("=" * 55)
    print(f"  Whisper Large-v3 + VAD + Diarization")
    print(f"  http://localhost:{PORT}")
    print("=" * 55)

    threading.Thread(target=_open_browser, daemon=True).start()
    threading.Thread(target=_auto_load,    daemon=True).start()

    app.run(debug=False, host='0.0.0.0', port=PORT,
            threaded=True, use_reloader=False)
