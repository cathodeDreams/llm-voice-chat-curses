from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List
import time
import numpy as np
from pathlib import Path

from .llama import LlamaChat
from .kokoro import Kokoro
from .config import AppConfig
import src.whisper as whisper

@dataclass
class ChatState:
    """Represents the current state of the chat session."""
    chat_history: List[str]
    scroll_offset: int
    recording_stream: Optional[object]
    recording_frames: Optional[List[bytes]]
    recording_pyaudio: Optional[object]
    last_user_message: Optional[str]
    processing: bool
    is_ptt_mode: bool
    sample_rate: Optional[int]

class ChatManager:
    def __init__(self, config: AppConfig, selected_settings: dict, status_callback: Callable[[str], None]):
        """Initialize chat manager with configuration and UI callback."""
        self.config = config
        self.settings = selected_settings
        self.status_callback = status_callback
        self.state = ChatState(
            chat_history=[],
            scroll_offset=0,
            recording_stream=None,
            recording_frames=None,
            recording_pyaudio=None,
            last_user_message=None,
            processing=False,
            is_ptt_mode=True,
            sample_rate=None
        )
        self.llm = None
        self.tts = None
        self.on_chat_updated = None  # Add callback reference

    def initialize_models(self) -> bool:
        """Initialize LLM and TTS models."""
        try:
            # Get paths
            selected_llm = self.settings.get("LLM Model")
            selected_prompt = self.settings.get("System Prompt", "template.xml")
            prompt_path = Path("prompts") / selected_prompt
            # Use the full model path directly without taking parent
            llm_model_path = Path(self.config.llm.model_path) / selected_llm

            # Initialize LLM
            self.status_callback(f"Loading LLM model: {selected_llm}...")
            print(f"Loading model from path: {llm_model_path} (exists: {llm_model_path.exists()})")  # Debug print
            self.llm = LlamaChat(
                model_path=str(llm_model_path),
                n_gpu_layers=self.config.llm.n_gpu_layers,
                n_ctx=self.config.llm.n_ctx,
                system_prompt_path=str(prompt_path),
                temperature=self.config.llm.temperature,
                top_p=self.config.llm.top_p,
                top_k=self.config.llm.top_k,
                repetition_penalty=self.config.llm.repetition_penalty,
                frequency_penalty=self.config.llm.frequency_penalty,
                presence_penalty=self.config.llm.presence_penalty
            )

            # Initialize TTS
            self.status_callback("Initializing TTS system...")
            self.tts = Kokoro(
                model_path=self.config.tts.model_path,
                voices_bin_path=self.config.tts.voices_bin_path
            )

            # Initialize Whisper
            self.status_callback("Loading Whisper speech recognition...")
            whisper.load_model("base")

            return True
        except Exception as e:
            self.status_callback(f"Error initializing models: {str(e)}")
            time.sleep(3)
            return False

    def process_voice_input(self, user_message: str) -> bool:
        """Process voice input and generate response."""
        if not user_message:
            return False

        self.state.last_user_message = user_message
        self.state.chat_history.append(f"You: {user_message}")
        if self.on_chat_updated:  # Immediately update UI with user message
            self.on_chat_updated()
        self.state.processing = True
        self.status_callback("Thinking...")

        try:
            # Generate response
            assistant_message = self.llm.chat(user_message)
            self.state.chat_history.append(f"Assistant: {assistant_message}")
            if self.on_chat_updated:  # Update UI with assistant message before TTS
                self.on_chat_updated()

            if self.state.processing:  # Check if we haven't been stopped
                self.status_callback("Synthesizing speech...")
                
                # Get voice settings
                selected_voice = self.settings.get("TTS Voice")
                if selected_voice == "[ Blend Voices ]":
                    voice1 = self.settings.get("Voice 1")
                    voice2 = self.settings.get("Voice 2")
                    ratio = self.settings.get("Blend Ratio", 50)
                    audio_data, sample_rate = self.tts.create_with_blend(
                        text=assistant_message,
                        voice1=voice1,
                        voice2=voice2,
                        blend_ratio=ratio/100,
                        speed=0.9,
                        lang="en-us"
                    )
                else:
                    audio_data, sample_rate = self.tts.create(
                        text=assistant_message,
                        voice=selected_voice,
                        speed=0.9,
                        lang="en-us"
                    )

                if audio_data is not None and self.state.processing:
                    self.status_callback("Playing response...")
                    whisper.play_audio(audio_data, sample_rate)

            self.state.processing = False
            self.status_callback("Ready to chat!")
            return True

        except Exception as e:
            self.status_callback(f"Error processing input: {str(e)}")
            self.state.processing = False
            return False

    def start_recording(self) -> bool:
        """Start audio recording."""
        if self.state.recording_stream is not None or self.state.processing:
            return False

        stream, frames, pyaudio, sample_rate = whisper.record_audio()
        if not stream:
            self.status_callback("Error: Could not start recording")
            time.sleep(2)
            return False

        self.state.recording_stream = stream
        self.state.recording_frames = frames
        self.state.recording_pyaudio = pyaudio
        self.state.sample_rate = sample_rate
        stream.start_stream()
        return True

    def stop_recording(self) -> Optional[Tuple[np.ndarray, int]]:
        """Stop recording and return audio data."""
        if self.state.recording_stream is None:
            return None

        self.status_callback("Processing audio...")
        audio_data = whisper.stop_recording(
            self.state.recording_stream,
            self.state.recording_frames,
            self.state.recording_pyaudio
        )
        sample_rate = self.state.sample_rate

        # Reset recording state
        self.state.recording_stream = None
        self.state.recording_frames = None
        self.state.recording_pyaudio = None
        self.state.sample_rate = None

        if audio_data is not None and len(audio_data) > 0:
            return audio_data, sample_rate
        return None

    def clear_history(self):
        """Clear chat history."""
        self.state.chat_history.clear()
        self.state.scroll_offset = 0
        self.status_callback("Chat history cleared!")
        if self.on_chat_updated:  # Notify UI to update
            self.on_chat_updated()

    def toggle_ptt_mode(self):
        """Toggle between PTT and passive mode."""
        self.state.is_ptt_mode = not self.state.is_ptt_mode
        mode_name = "Push-to-Talk" if self.state.is_ptt_mode else "Passive"
        self.status_callback(f"Switched to {mode_name} mode")

    def stop_processing(self):
        """Stop current processing."""
        if self.state.processing:
            self.state.processing = False
            self.status_callback("Processing stopped!")

    def redo_last_message(self) -> bool:
        """Redo the last user message."""
        if not self.state.last_user_message or self.state.processing:
            return False

        # Remove last exchange
        self.state.chat_history = self.state.chat_history[:-2] if len(self.state.chat_history) >= 2 else []
        if self.on_chat_updated:  # Notify UI to update
            self.on_chat_updated()
        return self.process_voice_input(self.state.last_user_message)

    def cleanup(self):
        """Clean up resources."""
        if self.state.recording_stream is not None:
            whisper.stop_recording(
                self.state.recording_stream,
                self.state.recording_frames,
                self.state.recording_pyaudio
            )

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Transcribe audio data to text."""
        self.status_callback("Transcribing speech...")
        return whisper.transcribe(audio_data, sample_rate)

    def handle_passive_recording(self) -> Optional[str]:
        """Handle passive recording mode."""
        def status_callback(status):
            self.status_callback(status)
            # Note: UI update is handled by the UI layer
        
        return whisper.passive_record_and_transcribe(status_callback) 