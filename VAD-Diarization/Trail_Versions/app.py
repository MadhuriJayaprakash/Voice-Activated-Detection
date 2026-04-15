from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
import whisper
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
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

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Fix network timeouts and large file issues
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

class SpeechAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = None
        self.diarization_pipeline = None
        self.vad_pipeline = None
        self.models_loaded = False
        self.loading_progress = 0
        self.loading_status = "Not started"
    
    def load_models_async(self, hf_token, whisper_model_size="base"):
        """Load models asynchronously to prevent timeout"""
        def load_models_thread():
            try:
                self.loading_status = "Loading Whisper model..."
                self.loading_progress = 10
                print(f"Loading models on {self.device}")
                
                # Load Whisper model with error handling
                print("🔄 Loading Whisper model...")
                self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
                self.loading_progress = 40
                print("✅ Whisper model loaded")
                
                # Load PyAnnote diarization pipeline
                self.loading_status = "Loading speaker diarization..."
                self.loading_progress = 60
                print("🔄 Loading speaker diarization...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                self.diarization_pipeline.to(self.device)
                print("✅ Diarization pipeline loaded")
                
                # Load VAD pipeline
                self.loading_status = "Loading Voice Activity Detection..."
                self.loading_progress = 80
                print("🔄 Loading Voice Activity Detection...")
                vad_model = Model.from_pretrained(
                    "pyannote/segmentation-3.0", 
                    use_auth_token=hf_token
                )
                self.vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
                self.vad_pipeline.instantiate({
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.3
                })
                
                self.loading_progress = 100
                self.loading_status = "Models loaded successfully"
                self.models_loaded = True
                print("✅ All models loaded successfully")
                
            except Exception as e:
                self.loading_status = f"Error: {str(e)}"
                self.loading_progress = 0
                print(f"❌ Model loading failed: {str(e)}")
                self.models_loaded = False
        
        # Start loading in background thread
        thread = threading.Thread(target=load_models_thread)
        thread.daemon = True
        thread.start()
        return True
    
    def get_loading_status(self):
        """Get current loading status"""
        return {
            'progress': self.loading_progress,
            'status': self.loading_status,
            'models_loaded': self.models_loaded
        }
    
    def preprocess_audio(self, audio_path, target_sr=16000):
        """Preprocess audio for better results"""
        try:
            # Load and normalize audio
            audio, sr = librosa.load(audio_path, sr=target_sr)
            audio = librosa.util.normalize(audio)
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Save preprocessed audio
            preprocessed_path = audio_path.replace('.wav', '_preprocessed.wav')
            sf.write(preprocessed_path, audio, target_sr)
            
            return preprocessed_path, len(audio) / target_sr
            
        except Exception as e:
            print(f"Audio preprocessing failed: {str(e)}")
            return None, None
    
    def detect_voice_activity(self, audio_path):
        """Detect speech and silence segments with energy validation"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # First check: Audio energy level
            audio_energy = torch.mean(torch.abs(waveform)).item()
            print(f"🔊 Audio energy level: {audio_energy:.6f}")
            
            # If audio is extremely quiet (likely silent), return empty
            if audio_energy < 0.001:
                print("⚠️ Audio energy too low - likely silent audio")
                return [], []
            
            vad_result = self.vad_pipeline({
                "waveform": waveform, 
                "sample_rate": sample_rate
            })
            
            speech_segments = []
            silence_segments = []
            
            # Extract speech segments with minimum duration filter
            for speech in vad_result.get_timeline():
                # Only include segments longer than 0.3 seconds
                if speech.duration >= 0.3:
                    speech_segments.append({
                        'type': 'speech',
                        'start': round(speech.start, 2),
                        'end': round(speech.end, 2),
                        'duration': round(speech.duration, 2)
                    })
            
            return speech_segments, silence_segments
            
        except Exception as e:
            print(f"VAD failed: {str(e)}")
            return [], []
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        try:
            # Transcribe with options to reduce hallucinations
            result = self.whisper_model.transcribe(
                audio_path,
                task='transcribe',
                language='en',
                word_timestamps=True,
                verbose=False,
                condition_on_previous_text=False,  # Reduce hallucinations
                compression_ratio_threshold=2.4,   # Filter out repetitive text
                logprob_threshold=-1.0,            # Filter low-confidence segments
                no_speech_threshold=0.6            # Better silence detection
            )
            
            return result
            
        except Exception as e:
            print(f"Transcription failed: {str(e)}")
            return None
    
    def perform_diarization(self, audio_path, min_speakers=None, max_speakers=None):
        """Perform speaker diarization with timeout handling"""
        try:
            kwargs = {}
            if min_speakers:
                kwargs['min_speakers'] = min_speakers
            if max_speakers:
                kwargs['max_speakers'] = max_speakers
            
            diarization = self.diarization_pipeline(audio_path, **kwargs)
            return diarization
            
        except Exception as e:
            print(f"Diarization failed: {str(e)}")
            return None
    
    def combine_transcript_and_diarization(self, transcript_result, diarization):
        """Combine transcription with speaker labels"""
        try:
            segments = transcript_result.get('segments', [])
            combined_segments = []
            
            for segment in segments:
                segment_start = segment['start']
                segment_end = segment['end']
                segment_duration = segment_end - segment_start
                segment_text = segment['text'].strip()
                
                # Skip very short segments (likely noise or hallucinations)
                if segment_duration < 0.5:
                    print(f"⚠️ Skipping short segment ({segment_duration:.1f}s): {segment_text[:50]}")
                    continue
                
                # Skip segments with very few words for their duration
                word_count = len(segment_text.split())
                if word_count < 2 and segment_duration < 2.0:
                    print(f"⚠️ Skipping low-content segment: {segment_text[:50]}")
                    continue
                
                # Find speaker for this segment
                speaker = 'Unknown'
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    if (segment_start >= turn.start and segment_start < turn.end) or \
                       (segment_end > turn.start and segment_end <= turn.end) or \
                       (segment_start <= turn.start and segment_end >= turn.end):
                        speaker = spk
                        break
                
                if segment_text:
                    combined_segments.append({
                        'speaker': speaker,
                        'start': round(segment_start, 2),
                        'end': round(segment_end, 2),
                        'duration': round(segment_duration, 2),
                        'text': segment_text,
                        'start_time': str(timedelta(seconds=int(segment_start))),
                        'end_time': str(timedelta(seconds=int(segment_end))),
                        'word_count': word_count
                    })
            
            return combined_segments
            
        except Exception as e:
            print(f"Combining transcript and diarization failed: {str(e)}")
            return []
    
    def analyze_audio(self, audio_path, min_speakers=None, max_speakers=None):
        """Complete audio analysis pipeline with progress updates"""
        if not self.models_loaded:
            return {'error': 'Models not loaded'}
        
        try:
            # Preprocess audio
            processed_path, duration = self.preprocess_audio(audio_path)
            if processed_path is None:
                return {'error': 'Audio preprocessing failed'}
            
            results = {'duration': duration}
            
            # Voice Activity Detection
            print("🔍 Detecting voice activity...")
            speech_segments, silence_segments = self.detect_voice_activity(processed_path)
            results['speech_segments'] = speech_segments
            results['silence_segments'] = silence_segments
            
            # Check if there's any actual speech
            total_speech_duration = sum(seg['duration'] for seg in speech_segments)
            speech_ratio = total_speech_duration / duration if duration > 0 else 0
            
            print(f"📊 Speech detected: {total_speech_duration:.2f}s / {duration:.2f}s ({speech_ratio*100:.1f}%)")
            
            # If less than 5% of audio contains speech, consider it silent
            if speech_ratio < 0.05 or total_speech_duration < 1.0:
                print("⚠️ No significant speech detected in audio")
                results['segments'] = []
                results['timeline'] = []
                results['unique_speakers_count'] = 0
                results['warning'] = 'No significant speech detected in this audio file'
                
                # Cleanup
                if os.path.exists(processed_path):
                    os.unlink(processed_path)
                
                return results
            
            # Transcription
            print("🎵 Transcribing audio...")
            transcript_result = self.transcribe_audio(processed_path)
            if transcript_result is None:
                return {'error': 'Transcription failed'}
            
            # Speaker Diarization
            print("👥 Performing speaker diarization...")
            diarization = self.perform_diarization(processed_path, min_speakers, max_speakers)
            if diarization is None:
                return {'error': 'Diarization failed'}
            
            # Combine results
            print("🏷️ Combining transcription with speakers...")
            combined_segments = self.combine_transcript_and_diarization(transcript_result, diarization)
            results['segments'] = combined_segments
            
            # Create timeline
            results['timeline'] = self.create_timeline(combined_segments, silence_segments, duration)
            
            # Calculate unique speakers
            unique_speakers = set(seg['speaker'] for seg in combined_segments if seg['speaker'] != 'Unknown')
            results['unique_speakers_count'] = len(unique_speakers) if unique_speakers else 0
            
            print(f"✅ Analysis complete: {len(combined_segments)} segments, {results['unique_speakers_count']} speakers")
            
            # Cleanup
            if os.path.exists(processed_path):
                os.unlink(processed_path)
            
            # Memory cleanup
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def create_timeline(self, segments, silence_segments, duration):
        """Create a combined timeline of speech and silence events"""
        all_events = []
        
        # Add speech segments
        for segment in segments:
            all_events.append({
                'start': segment['start'],
                'end': segment['end'],
                'type': 'speech',
                'speaker': segment['speaker'],
                'text': segment['text'],
                'duration': segment['duration']
            })
        
        # Add significant silence segments
        for silence in silence_segments:
            if silence['duration'] >= 0.5:
                all_events.append({
                    'start': silence['start'],
                    'end': silence['end'],
                    'type': 'silence',
                    'speaker': None,
                    'text': '[SILENCE]',
                    'duration': silence['duration']
                })
        
        # Sort by start time
        all_events.sort(key=lambda x: x['start'])
        return all_events

# Global analyzer instance
analyzer = SpeechAnalyzer()

# Auto-load models if HF_TOKEN is set in .env
def auto_load_models():
    """Automatically load models if HF_TOKEN is configured in .env"""
    hf_token = os.getenv('HF_TOKEN', '').strip()
    
    if hf_token and not analyzer.models_loaded:
        print("🔑 HF_TOKEN found in .env file")
        print("🚀 Auto-loading models with Whisper model: base")
        analyzer.load_models_async(hf_token, 'base')
    elif not hf_token:
        print("ℹ️  No HF_TOKEN in .env - add your token to .env file")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Error handlers for network issues
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413

@app.errorhandler(500)
def handle_internal_error(error):
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.route('/')
def index():
    device_info = {
        'device': analyzer.device_str.upper(),
        'cuda_available': torch.cuda.is_available(),
        'models_loaded': analyzer.models_loaded,
        'has_env_token': bool(os.getenv('HF_TOKEN', '').strip())
    }
    return render_template('index.html', device_info=device_info)

@app.route('/load_models', methods=['POST', 'OPTIONS'])
def load_models():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get token from .env only
        hf_token = os.getenv('HF_TOKEN', '').strip()
        
        # Get whisper model from form
        if request.is_json:
            whisper_model = request.json.get('whisper_model', 'base')
        else:
            whisper_model = request.form.get('whisper_model', 'base')
        
        print(f"Using token from .env: {'***' + hf_token[-4:] if hf_token and len(hf_token) > 4 else 'None'}")
        print(f"Whisper model: {whisper_model}")
        
        if not hf_token:
            return jsonify({'success': False, 'error': 'HF_TOKEN not found in .env file'}), 400
        
        # Start async loading
        success = analyzer.load_models_async(hf_token, whisper_model)
        if success:
            return jsonify({'success': True, 'message': 'Model loading started'}), 200
        else:
            return jsonify({'success': False, 'error': 'Failed to start model loading'}), 500
            
    except Exception as e:
        print(f"Error in load_models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/loading_status', methods=['GET'])
def loading_status():
    """Get current model loading status"""
    status = analyzer.get_loading_status()
    return jsonify(status)

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
        return jsonify({'error': 'Invalid file format'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Convert to WAV if needed
        if not filename.lower().endswith('.wav'):
            temp_path = filepath
            file.save(temp_path)
            
            # Convert to WAV
            audio = AudioSegment.from_file(temp_path)
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format='wav')
            
            os.unlink(temp_path)
            filepath = wav_path
        else:
            file.save(filepath)
        
        # Get analysis parameters
        min_speakers = request.form.get('min_speakers', type=int)
        max_speakers = request.form.get('max_speakers', type=int)
        
        # Analyze audio
        results = analyzer.analyze_audio(filepath, min_speakers, max_speakers)
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 500
        
        results['filename'] = filename
        return render_template('results.html', results=results)
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    finally:
        # Cleanup uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except:
                pass

if __name__ == '__main__':
    # Don't auto-load - let user select model in UI
    # auto_load_models()
    
    # Run with better network settings
    app.run(
        debug=True, 
        host='0.0.0.0',  # Allow external connections
        port=5000,
        threaded=True,   # Handle multiple requests
        use_reloader=False  # Prevent model reloading
    )
