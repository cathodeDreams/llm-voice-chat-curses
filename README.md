# LLM Voice Chat

Voice chat interface combining a local [GGUF](https://huggingface.co/docs/gguf/en/index) LLM, [Whisper](https://github.com/openai/whisper) for speech recognition, and [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) for natural AI conversations. Features push-to-talk and passive listening modes.

## Features

- Real-time voice input with Whisper
- Local GGUF LLM-based chat with GPU acceleration
- Kokoro TTS with voice blending
- Terminal UI with curses
- Configurable system prompts

## Requirements

- Python 3.8+
- A compatible CPU, or an NVIDIA GPU with CUDA support.
- CUDA Toolkit 11.8+ (if using NVIDIA GPU)
- Microphone and audio output

## Installation

This guide provides general instructions.  Specific commands may vary slightly depending on your distribution.

### 1. Clone the Repository

```bash
git clone https://github.com/cathodDreams/llm-voice-chat-curses.git
cd llm_voice_chat
```

### 2. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

**Activate the environment:**

*   **Linux/macOS:**

    ```bash
    source venv/bin/activate
    ```
*   **Windows (cmd.exe):**

    ```cmd
    venv\Scripts\activate.bat
    ```
*   **Windows (PowerShell):**

    ```powershell
    .\venv\Scripts\Activate.ps1
    ```

### 3. Install Dependencies

The installation process differs slightly depending on whether you're using CPU or GPU acceleration.

#### **Option A: CPU-Only Installation (All Platforms)**

This is the simplest option and works on all systems.

```bash
pip install -r requirements.txt
```

#### **Option B: GPU Acceleration (NVIDIA, Linux)**

This provides significantly faster LLM inference if you have a compatible NVIDIA GPU.

1.  **Install CUDA Toolkit 11.8+:**  Make sure you have the correct CUDA Toolkit installed and that your NVIDIA drivers are up-to-date.  You can usually find the appropriate drivers and toolkit on the NVIDIA website.

2.  **Install `llama-cpp-python` with CUDA support:**

    The following command forces a reinstall of `llama-cpp-python` with the necessary CUDA flags.  This ensures it's built against your installed CUDA Toolkit.

    ```bash
    CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_CUDA_F16=1" pip install --force-reinstall --no-cache-dir llama-cpp-python
    ```

3.  **Install other requirements:**

    ```bash
    pip install -r requirements.txt
    ```

#### **Option C: GPU Acceleration (NVIDIA, Windows)**

1.  **Install CUDA Toolkit 11.8+:** Download and install the appropriate CUDA Toolkit for your system from the NVIDIA website.  Ensure your NVIDIA drivers are also up-to-date.

2.  **Install `llama-cpp-python` with CUDA support:**

    Open a command prompt (cmd.exe) or PowerShell *as administrator*.  Then, use the following command:

    *   **cmd.exe:**

        ```cmd
        set CMAKE_ARGS=-DGGML_CUDA=on -DLLAMA_CUDA_F16=1
        pip install --force-reinstall --no-cache-dir llama-cpp-python
        ```

    *   **PowerShell:**

        ```powershell
        $env:CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_CUDA_F16=1"
        pip install --force-reinstall --no-cache-dir llama-cpp-python
        ```

3.  **Install other requirements:**

    ```bash
    pip install -r requirements.txt
    ```
    
#### **Option D: GPU Acceleration (Arch Linux)**
1. Install CUDA, llama-cpp-python, and other requirements:
    ```bash
    pacman -Syu --needed cuda python-pip python-venv
    pip install --force-reinstall --no-cache-dir llama-cpp-python
    pip install -r requirements.txt
    ```

### 4. Setup Models

1.  **LLM Model:**
    *   Download a GGUF-format LLM model.  You can find many models on [Hugging Face](https://huggingface.co/models?search=gguf).  A good starting point is a 7B parameter model.
    *   Place the downloaded `.gguf` file in the `models/llm/` directory.

2.  **Kokoro TTS Model:**
    *   Download the `kokoro-v1.0.onnx` and `voices-v1.0.bin` files from the [Kokoro-onnx releases page](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0).
    *   Place these files in the `models/tts/` directory.
    *   Create a `voices.txt` file in the `models/tts` directory. You can copy the contents from [Kokoro-82M/VOICES.md](https://github.com/thewh1teagle/kokoro-onnx/blob/main/Kokoro-82M/VOICES.md) and paste them into your `voices.txt` file.  This file defines the available voices.

### 5. Configure

1.  **`config.yaml`:**
    *   Adjust the paths in `config.yaml` to point to your downloaded LLM and TTS models.  The default paths should work if you placed the models in the recommended directories.
    *   Modify other parameters like `n_gpu_layers`, `temperature`, etc., as desired.  `n_gpu_layers` controls how much of the LLM is offloaded to the GPU (higher values = more GPU usage).  If you have a powerful GPU, you can increase this value.
    *   The `colors` section allows you to customize the terminal UI colors.

2.  **System Prompt (`prompts/template.xml`):**
    *   The `prompts/` directory contains XML files that define the system prompt for the LLM.  You can create different prompts and select them in the UI.  The `template.xml` file is a good starting point.

## Usage

1.  **Activate your virtual environment** (if you created one):

    *   **Linux/macOS:**  `source venv/bin/activate`
    *   **Windows:**  `venv\Scripts\activate` (or `venv\Scripts\Activate.ps1`)

2.  **Run the application:**

    ```bash
    python main.py
    ```

3.  **Controls:**

    *   **Spacebar:**  Push-to-talk (when in PTT mode)
    *   **ESC:**  Exit the application
    *   **Arrow Keys (Up/Down):** Scroll through the chat history
    *   **Mouse:** Click on buttons in the UI

## Troubleshooting

*   **`llama_cpp_python` installation issues:**  If you have trouble with `llama-cpp-python`, especially with GPU support, make sure you have the correct CUDA Toolkit version installed and that you're using the correct `CMAKE_ARGS` during installation.  You may need to consult the `llama-cpp-python` documentation for more specific instructions.
*   **Model loading errors:** Double-check the paths in `config.yaml` and ensure that the model files are in the correct locations.
*   **Audio issues:** Verify that your microphone and speakers are correctly configured in your operating system.
* **Kokoro TTS errors:** Ensure that `kokoro-v1.0.onnx` and `voices-v1.0.bin` are in the `models/tts` directory, and that `voices.txt` is correctly populated.

## Configuration

### LLM Settings (config.yaml)

-   `llm_model_path`: Path to the directory containing your GGUF LLM model.
-   `n_gpu_layers`: Number of layers to offload to the GPU (if available).
-   `n_ctx`: Context window size.
-   `temperature`: Controls the randomness of the generated text (0.0 is deterministic, 1.0 is highly random).
-   `top_p`, `top_k`: Sampling parameters that affect the diversity of the generated text.
-   `repetition_penalty`:  Discourages the model from repeating the same phrases.
-   `frequency_penalty`: Decreases the likelihood of common tokens.
-   `presence_penalty`:  Encourages the model to use new tokens.

### TTS Settings (config.yaml)

-   `tts_model_path`: Path to the Kokoro ONNX model file.
-   `tts_voices_path`: Path to the `voices.txt` file.
-   `voices_path`: Path to the `voices-v1.0.bin` file.

### UI Customization (config.yaml)

-   The `colors` section allows you to customize the colors of the UI elements.  You can use standard curses color names (BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, DEFAULT).

### System Prompt (prompts/template.xml)

- The system prompt instructs the LLM on its role and behavior. You can create multiple prompt files in the `prompts/` directory and select them in the UI.

---

*Note: This project was conceived by Azul and executed with assistance from Claude 3.5 Sonnet, Gemini 2.0 Pro, o3 mini, and Deepseek R1. The text you are reading now was generated by Claude 3.5 Sonnet and updated by Gemini 2.0 Pro.*
