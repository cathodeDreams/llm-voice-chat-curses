from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import yaml
from pathlib import Path

@dataclass
class LLMConfig:
    model_path: str
    n_gpu_layers: int
    n_ctx: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float

@dataclass
class TTSConfig:
    model_path: str
    voices_path: str
    voices_bin_path: str

@dataclass
class ColorConfig:
    menu_selected: Tuple[str, str]
    recording_status: Tuple[str, str]
    ready_status: Tuple[str, str]
    user_message: Tuple[str, str]
    assistant_message: Tuple[str, str]

@dataclass
class AppConfig:
    llm: LLMConfig
    tts: TTSConfig
    colors: ColorConfig
    system_prompt_path: str

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.workspace_root = Path(__file__).parent.parent  # Go up one level from src/
        self.config_path = self.workspace_root / config_path
        self.config = self._load_config()

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the workspace root."""
        return self.workspace_root / path

    def _load_config(self) -> AppConfig:
        """Load and validate configuration from yaml file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            print(f"Loading config from: {self.config_path}")  # Debug print
            print(f"Workspace root: {self.workspace_root}")  # Debug print

            # Create LLM config
            llm_path = self._resolve_path(config_data.get('llm_model_path', 'models/llm'))
            print(f"Resolved LLM path: {llm_path} (exists: {llm_path.exists()})")  # Debug print
            
            llm_config = LLMConfig(
                model_path=str(llm_path),
                n_gpu_layers=config_data.get('n_gpu_layers', 20),
                n_ctx=config_data.get('n_ctx', 4096),
                temperature=config_data.get('temperature', 0.3),
                top_p=config_data.get('top_p', 1.0),
                top_k=config_data.get('top_k', 40),
                repetition_penalty=config_data.get('repetition_penalty', 1.0),
                frequency_penalty=config_data.get('frequency_penalty', 0.0),
                presence_penalty=config_data.get('presence_penalty', 0.0)
            )

            # Create TTS config
            tts_config = TTSConfig(
                model_path=str(self._resolve_path(config_data.get('tts_model_path', 'models/tts/kokoro-v1.0.onnx'))),
                voices_path=str(self._resolve_path(config_data.get('tts_voices_path', 'models/tts/voices.txt'))),
                voices_bin_path=str(self._resolve_path(config_data.get('voices_path', 'models/tts/voices-v1.0.bin')))
            )

            # Create color config
            colors = config_data.get('colors', {})
            color_config = ColorConfig(
                menu_selected=self._get_color_pair(colors.get('menu_selected', {})),
                recording_status=self._get_color_pair(colors.get('recording_status', {})),
                ready_status=self._get_color_pair(colors.get('ready_status', {})),
                user_message=self._get_color_pair(colors.get('user_message', {})),
                assistant_message=self._get_color_pair(colors.get('assistant_message', {}))
            )

            return AppConfig(
                llm=llm_config,
                tts=tts_config,
                colors=color_config,
                system_prompt_path=str(self._resolve_path(config_data.get('system_prompt_path', 'prompts/template.xml')))
            )

        except Exception as e:
            raise ConfigError(f"Error loading configuration: {str(e)}")

    def _get_color_pair(self, color_dict: Dict[str, str]) -> Tuple[str, str]:
        """Extract foreground and background colors from color dictionary."""
        return (
            color_dict.get('foreground', 'WHITE'),
            color_dict.get('background', 'DEFAULT')
        )

    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config

    def validate_paths(self) -> bool:
        """Validate that all required files and directories exist."""
        paths_to_check = [
            Path(self.config.llm.model_path),
            Path(self.config.tts.model_path),
            Path(self.config.tts.voices_path),
            Path(self.config.tts.voices_bin_path),
            Path(self.config.system_prompt_path)
        ]

        missing_paths = []
        for path in paths_to_check:
            if not path.exists():
                missing_paths.append(str(path))

        if missing_paths:
            raise ConfigError(f"Missing required files: {', '.join(missing_paths)}")

        return True

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass 