# Placeholder functions for Whisper integration.  You'll need to replace
# these with your actual Whisper implementation.
import pyaudio
import wave
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import time

_model = None  # Global variable to hold the loaded model

def load_model(model_name="base"):
    """Loads the Whisper model. Keeps it in memory for later use."""
    global _model
    if _model is None:
        try:
            _model = whisper.load_model(model_name)
            print(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise  # Re-raise the exception to stop execution if loading fails
    return _model

def record_audio(sample_rate=16000, channels=1):  # Changed default sample rate to match Whisper's expected rate
    """Records audio until stopped. Returns the audio data directly."""
    try:
        p = pyaudio.PyAudio()
        frames = []

        def callback(in_data, frame_count, time_info, status):
            if status:
                print(f"PyAudio callback status: {status}")
            frames.append(in_data)
            return (in_data, pyaudio.paContinue)

        # Use float32 format for better compatibility with Whisper
        stream = p.open(format=pyaudio.paFloat32,
                       channels=channels,
                       rate=sample_rate,
                       input=True,
                       frames_per_buffer=1024,
                       stream_callback=callback,
                       start=False)  # Don't start automatically

        return stream, frames, p, sample_rate

    except Exception as e:
        print(f"Error during recording setup: {e}")
        if 'p' in locals():
            p.terminate()
        return None, None, None, None

def stop_recording(stream, frames, p):
    """Stops the recording and returns the audio data."""
    try:
        if stream is None or p is None:
            return None

        stream.stop_stream()
        stream.close()
        p.terminate()

        if not frames:  # Check if we have any recorded data
            print("No audio data recorded")
            return None

        # Convert frames to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        
        # Ensure audio is in the correct range for Whisper (-1 to 1)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data

    except Exception as e:
        print(f"Error stopping recording: {e}")
        return None

def transcribe(audio_data: np.ndarray, sample_rate: int) -> str:
    """Transcribes the given audio data using the loaded Whisper model."""
    global _model
    if _model is None:
        load_model()

    try:
        if audio_data is None or len(audio_data) == 0:
            print("No audio data to transcribe")
            return ""

        # Ensure audio is mono for Whisper
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed (Whisper's required sample rate)
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        result = _model.transcribe(audio_data)
        return result["text"].strip()

    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def play_audio(audio_data: np.ndarray, sample_rate: int):
    """Plays audio data directly from memory."""
    try:
        if audio_data is None or len(audio_data) == 0:
            print("No audio data to play")
            return

        # Ensure audio data is in float32 format
        audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error during audio playback: {e}")

def passive_record_and_transcribe(status_update_callback=None) -> str:
    """
    Records audio passively, waiting for speech and stopping after silence.
    Returns the transcribed text.
    """
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCKSIZE = 1024
    DTYPE = np.float32
    
    audio_data = []
    silence_threshold = 0.01
    silence_duration_required = 3.0
    accumulated_silence_time = 0.0
    check_interval = 0.5

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        audio_data.extend(indata.flatten())

    if status_update_callback:
        status_update_callback("⌛ Waiting before recording...")
    time.sleep(1)

    if status_update_callback:
        status_update_callback("🎤 Recording...")

    try:
        with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, 
                          dtype=DTYPE, blocksize=BLOCKSIZE, callback=callback):
            # Wait for initial audio activity
            initial_audio_detected = False
            while not initial_audio_detected:
                time.sleep(check_interval)
                if len(audio_data) >= int(SAMPLE_RATE * check_interval):
                    recent_audio = np.array(audio_data[-int(SAMPLE_RATE * check_interval):])
                    rms = np.sqrt(np.mean(np.square(recent_audio)))
                    if rms > silence_threshold:
                        initial_audio_detected = True

            # Continue recording until silence
            while True:
                time.sleep(check_interval)
                if len(audio_data) >= int(SAMPLE_RATE * check_interval):
                    recent_audio = np.array(audio_data[-int(SAMPLE_RATE * check_interval):])
                    rms = np.sqrt(np.mean(np.square(recent_audio)))
                    if rms < silence_threshold:
                        accumulated_silence_time += check_interval
                    else:
                        accumulated_silence_time = 0.3
                if accumulated_silence_time >= silence_duration_required:
                    break

        if status_update_callback:
            status_update_callback("⏳ Transcribing...")

        if len(audio_data) > 0:
            audio_data = np.array(audio_data)
            result = transcribe(audio_data, SAMPLE_RATE)
            if status_update_callback:
                status_update_callback(f"Transcribed: {result}")
            return result
        return ""

    except Exception as e:
        print(f"Error in passive recording: {e}")
        return ""
