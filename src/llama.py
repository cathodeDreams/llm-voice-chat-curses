from llama_cpp import Llama
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

class LlamaChat:
    def __init__(self, model_path: str, n_gpu_layers: int, n_ctx: int, system_prompt_path: str, 
                 temperature: float = 1.5, top_p: float = 1.0, top_k: int = 40,
                 repetition_penalty: float = 1.0, frequency_penalty: float = 0.0, 
                 presence_penalty: float = 0.0):
        self.model_path = Path(model_path)
        self.llm = Llama(model_path=str(self.model_path), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        # Store LLM parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def _load_system_prompt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                prompt_name = Path(file_path).stem
                today_date = datetime.now().strftime("%Y-%m-%d")
                return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: {today_date}\n\nPrompt Name: {prompt_name}\n\n{content}<|eot_id|>"
        except Exception as e:
            return "You are a helpful assistant."

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        try:
            response = self.llm.create_chat_completion(
                messages=self.messages,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repetition_penalty,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            assistant_message = response['choices'][0]['message']['content']
            self.messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            return "An error occurred during the conversation."
