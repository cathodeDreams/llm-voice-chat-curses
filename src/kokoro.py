from dataclasses import dataclass
import sounddevice as sd
from kokoro_onnx import Kokoro as KokoroOnnx
import numpy as np
import os

@dataclass
class KokoroConfig:
    model_path: str
    voices_path: str
    use_gpu: bool = True  # Default to True since we have NVIDIA GPU

class Kokoro:
    def __init__(self, model_path: str, voices_bin_path: str):
        """Initialize Kokoro TTS with the given model and voices."""
        self.config = KokoroConfig(
            model_path=model_path,
            voices_path=voices_bin_path
        )
        
        # Set ONNX Runtime environment variables for GPU
        if self.config.use_gpu:
            os.environ["ONNXRUNTIME_PROVIDER_PATH"] = "/usr/lib/onnxruntime/providers"
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_MAX_PARTITION_ITERATIONS"] = "1000"
            os.environ["ORT_TENSORRT_MIN_SUBGRAPH_SIZE"] = "1"
        
        self.kokoro = KokoroOnnx(
            model_path=self.config.model_path,
            voices_path=self.config.voices_path
        )

    def create(self, text: str, voice: str, speed: float = 0.9, lang: str = "en-us") -> tuple[np.ndarray, int]:
        """
        Create speech from text using the specified voice.
        
        Args:
            text (str): The text to convert to speech
            voice (str): The voice ID or name to use
            speed (float, optional): Speech speed. Defaults to 0.9.
            lang (str, optional): Language code. Defaults to "en-us".
            
        Returns:
            tuple[np.ndarray, int]: Audio samples and sample rate
        """
        try:
            samples, sample_rate = self.kokoro.create(
                text=text,
                voice=voice,
                speed=speed,
                lang=lang
            )
            return samples, sample_rate
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None, None

    def get_voice_style(self, voice_name: str) -> np.ndarray:
        """
        Get the voice style array for a given voice name.
        Useful for voice blending.
        
        Args:
            voice_name (str): Name of the voice to get style for
            
        Returns:
            np.ndarray: Voice style array
        """
        try:
            return self.kokoro.get_voice_style(voice_name)
        except Exception as e:
            print(f"Error getting voice style: {e}")
            return None

    def create_with_blend(self, text: str, voice1: str, voice2: str, blend_ratio: float = 0.5, 
                         speed: float = 0.9, lang: str = "en-us") -> tuple[np.ndarray, int]:
        """
        Create speech using a blend of two voices.
        
        Args:
            text (str): The text to convert to speech
            voice1 (str): First voice name
            voice2 (str): Second voice name
            blend_ratio (float): Ratio of first voice (0.0 to 1.0)
            speed (float, optional): Speech speed. Defaults to 0.9.
            lang (str, optional): Language code. Defaults to "en-us".
            
        Returns:
            tuple[np.ndarray, int]: Audio samples and sample rate
        """
        try:
            # Get voice styles
            style1 = self.get_voice_style(voice1)
            style2 = self.get_voice_style(voice2)
            
            if style1 is None or style2 is None:
                return None, None
            
            # Blend voice styles
            blended_style = np.add(
                style1 * blend_ratio,
                style2 * (1.0 - blend_ratio)
            )
            
            # Create speech with blended style
            samples, sample_rate = self.kokoro.create(
                text=text,
                voice=blended_style,  # Pass the blended style directly
                speed=speed,
                lang=lang
            )
            return samples, sample_rate
            
        except Exception as e:
            print(f"Error generating blended speech: {e}")
            return None, None
