from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
import os
from datetime import timedelta
import librosa
import soundfile as sf
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import gc
import warnings
import threading
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000
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


class VoiceDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.vad_pipeline = None
        self.models_loaded = False
        self.loading_progress = 0
        self.loading_status = "Not started"
    
    def load_models_async(self, hf_token):
        """Load only VAD model"""
        def load_models_thread():
            try:
                self.loading_status = "Loading Voice Activity Detection..."
                self.loading_progress = 30
                print(f"🔄 Loading VAD model on {self.device}")
                
                vad_model = Model.from_pretrained(
                    "pyannote/segmentation-3.0", 
                    use_auth_token=hf_token
                )
                
                self.loading_progress = 70
                
                self.vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
                self.vad_pipeline.instantiate({
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.3
                })
                
                self.loading_progress = 100
                self.loading_status = "VAD model loaded successfully"
                self.models_loaded = True
                print("✅ VAD model loaded successfully")
                
            except Exception as e:
                self.loading_status = f"Error: {str(e)}"
                self.loading_progress = 0
                print(f"❌ VAD loading failed: {str(e)}")
                self.models_loaded = False
        
        thread = threading.Thread(target=load_models_thread)
        thread.daemon = True
        thread.start()
        return True
    
    def get_loading_status(self):
        return {
            'progress': self.loading_progress,
            'status': self.loading_status,
            'models_loaded': self.models_loaded
        }
    
    def preprocess_audio(self, audio_path, target_sr=16000):
        """Preprocess audio"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            audio = librosa.util.normalize(audio)
            
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            preprocessed_path = audio_path.replace('.wav', '_preprocessed.wav')
            sf.write(preprocessed_path, audio, target_sr)
            
            return preprocessed_path, len(audio) / target_sr
            
        except Exception as e:
            print(f"Audio preprocessing failed: {str(e)}")
            return None, None
    
    def detect_voice_activity(self, audio_path):
        """Detect voice activity - speech and silence segments"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Check audio energy
            audio_energy = torch.mean(torch.abs(waveform)).item()
            print(f"🔊 Audio energy level: {audio_energy:.6f}")
            
            # If audio is extremely quiet
            if audio_energy < 0.001:
                print("⚠️ Audio energy too low - likely silent audio")
                return [], [], 0
            
            vad_result = self.vad_pipeline({
                "waveform": waveform, 
                "sample_rate": sample_rate
            })
            
            speech_segments = []
            total_speech_duration = 0
            
            # Extract speech segments
            for speech in vad_result.get_timeline():
                if speech.duration >= 0.3:  # Filter out very short segments
                    speech_segments.append({
                        'type': 'speech',
                        'start': round(speech.start, 2),
                        'end': round(speech.end, 2),
                        'duration': round(speech.duration, 2),
                        'start_time': str(timedelta(seconds=int(speech.start))),
                        'end_time': str(timedelta(seconds=int(speech.end)))
                    })
                    total_speech_duration += speech.duration
            
            # Calculate silence segments
            audio_duration = waveform.shape[1] / sample_rate
            silence_segments = []
            
            if speech_segments:
                # Silence before first speech
                if speech_segments[0]['start'] > 0.1:
                    silence_segments.append({
                        'type': 'silence',
                        'start': 0,
                        'end': speech_segments[0]['start'],
                        'duration': round(speech_segments[0]['start'], 2)
                    })
                
                # Silence between speeches
                for i in range(len(speech_segments) - 1):
                    silence_start = speech_segments[i]['end']
                    silence_end = speech_segments[i + 1]['start']
                    silence_duration = silence_end - silence_start
                    
                    if silence_duration > 0.3:
                        silence_segments.append({
                            'type': 'silence',
                            'start': round(silence_start, 2),
                            'end': round(silence_end, 2),
                            'duration': round(silence_duration, 2)
                        })
                
                # Silence after last speech
                if speech_segments[-1]['end'] < audio_duration - 0.1:
                    silence_segments.append({
                        'type': 'silence',
                        'start': speech_segments[-1]['end'],
                        'end': round(audio_duration, 2),
                        'duration': round(audio_duration - speech_segments[-1]['end'], 2)
                    })
            
            return speech_segments, silence_segments, total_speech_duration
            
        except Exception as e:
            print(f"VAD failed: {str(e)}")
            return [], [], 0
    
    def analyze_audio(self, audio_path):
        """VAD analysis only"""
        if not self.models_loaded:
            return {'error': 'VAD model not loaded'}
        
        try:
            # Preprocess audio
            print("🎵 Preprocessing audio...")
            processed_path, duration = self.preprocess_audio(audio_path)
            if processed_path is None:
                return {'error': 'Audio preprocessing failed'}
            
            results = {'duration': duration}
            
            # Voice Activity Detection
            print("🔍 Detecting voice activity...")
            speech_segments, silence_segments, total_speech = self.detect_voice_activity(processed_path)
            
            results['speech_segments'] = speech_segments
            results['silence_segments'] = silence_segments
            
            # Calculate speech ratio
            speech_ratio = total_speech / duration if duration > 0 else 0
            
            print(f"📊 Speech: {total_speech:.2f}s / {duration:.2f}s ({speech_ratio*100:.1f}%)")
            
            # If less than 5% speech detected
            if speech_ratio < 0.05 or total_speech < 1.0:
                print("⚠️ No significant speech detected")
                results['segments'] = []
                results['timeline'] = []
                results['unique_speakers_count'] = 0
                results['warning'] = 'No significant speech detected in this audio file'
            else:
                # Create segments for your existing template
                results['segments'] = []
                for i, seg in enumerate(speech_segments):
                    results['segments'].append({
                        'speaker': 'Voice Detected',
                        'start': seg['start'],
                        'end': seg['end'],
                        'duration': seg['duration'],
                        'text': f'[Speech segment {i+1}]',
                        'start_time': seg['start_time'],
                        'end_time': seg['end_time'],
                        'word_count': 0
                    })
                
                # Timeline
                results['timeline'] = self.create_timeline(speech_segments, silence_segments, duration)
                results['unique_speakers_count'] = 1 if speech_segments else 0
            
            print(f"✅ VAD complete: {len(speech_segments)} speech segments")
            
            # Cleanup
            if os.path.exists(processed_path):
                os.unlink(processed_path)
            
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def create_timeline(self, speech_segments, silence_segments, duration):
        """Create timeline for visualization"""
        all_events = []
        
        for i, segment in enumerate(speech_segments):
            all_events.append({
                'start': segment['start'],
                'end': segment['end'],
                'type': 'speech',
                'speaker': 'Voice Detected',
                'text': f'[Speech {i+1}]',
                'duration': segment['duration']
            })
        
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
        
        all_events.sort(key=lambda x: x['start'])
        return all_events


# Global detector
detector = VoiceDetector()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        'device': detector.device_str.upper(),
        'cuda_available': torch.cuda.is_available(),
        'models_loaded': detector.models_loaded,
        'has_env_token': bool(os.getenv('HF_TOKEN', '').strip())
    }
    return render_template('index.html', device_info=device_info)


@app.route('/load_models', methods=['POST', 'OPTIONS'])
def load_models():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        hf_token = os.getenv('HF_TOKEN', '').strip()
        
        if not hf_token:
            return jsonify({'success': False, 'error': 'HF_TOKEN not found in .env file'}), 400
        
        success = detector.load_models_async(hf_token)
        if success:
            return jsonify({'success': True, 'message': 'VAD model loading started'}), 200
        else:
            return jsonify({'success': False, 'error': 'Failed to start loading'}), 500
            
    except Exception as e:
        print(f"Error in load_models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/loading_status', methods=['GET'])
def loading_status():
    status = detector.get_loading_status()
    return jsonify(status)


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 200
    
    if not detector.models_loaded:
        return jsonify({'error': 'VAD model not loaded yet'}), 400
    
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
            
            audio = AudioSegment.from_file(temp_path)
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format='wav')
            
            os.unlink(temp_path)
            filepath = wav_path
        else:
            file.save(filepath)
        
        # VAD Analysis only
        results = detector.analyze_audio(filepath)
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 500
        
        results['filename'] = filename
        return render_template('results.html', results=results)
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except:
                pass


if __name__ == '__main__':
    app.run(
        debug=True, 
        host='0.0.0.0',
        port=5000,
        threaded=True,
        use_reloader=False
    )
