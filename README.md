# LLM Voice Chat

A voice-based chat interface that combines LLaMA, Whisper, and Kokoro TTS for natural conversation with AI. Features both push-to-talk and passive listening modes, with support for voice blending and GPU acceleration.

## Features

- Real-time voice input using Whisper for speech recognition
- LLaMA-based chat responses with configurable parameters
- High-quality text-to-speech using Kokoro TTS
- Voice blending capabilities for customized TTS voices
- Push-to-talk and passive listening modes
- Full GPU acceleration support
- Curses-based TUI (Terminal User Interface)
- Configurable system prompts using XML templates

## System Requirements

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- CUDA Toolkit 11.8 or higher
- At least 8GB of RAM (16GB recommended)
- Microphone for voice input
- Speakers or headphones for audio output

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cathodDreams/llm-voice-chat-curses.git
cd llm_voice_chat
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

3. Install CUDA dependencies (if not already installed):
```bash
# For Ubuntu/Debian:
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit
# For Arch Linux:
sudo pacman -S cuda
```

4. Install Python dependencies with CUDA support:
```bash
# Set environment variable for CUDA support
export GGML_CUDA=1
# Install requirements
pip install -r requirements.txt
```

5. Download required models:
   - Place your LLaMA model (GGUF format) in `models/llm/`
   - Place Kokoro TTS model in `models/tts/`
   - Place voice files in `models/tts/`

6. Configure the application:
   - Review and modify `config.yaml` as needed
   - Adjust system prompts in `prompts/template.xml`

## Usage

1. Start the application:
```bash
python main.py
```

2. In the main menu:
   - Select your LLM model
   - Choose a TTS voice (or set up voice blending)
   - Select a system prompt template
   - Click "Start Chat" or press Enter

3. During chat:
   - Push-to-talk mode: Hold SPACE to record, release to process
   - Passive mode: Automatically detects speech and processes it
   - Use mouse or keyboard shortcuts for navigation
   - ESC to exit
   - Arrow keys to scroll chat history

## Configuration

### LLM Settings (config.yaml)
- `n_gpu_layers`: Number of layers to offload to GPU
- `n_ctx`: Context window size
- `temperature`: Response randomness (0.0-1.0)
- `top_p`, `top_k`: Sampling parameters
- `repetition_penalty`: Penalty for repeated tokens

### TTS Settings
- Voice selection
- Voice blending ratios
- Speech speed
- Language settings

### UI Customization
- Color schemes
- Status indicators
- Button layouts 
