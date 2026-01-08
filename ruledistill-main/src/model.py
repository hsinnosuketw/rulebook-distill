from openai import OpenAI
import config
import prompt

class NVIDIAClient:
    """Implementation for NVIDIA NIM (OpenAI-compatible)"""
    def __init__(self):
        self.client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY
        )

    def call(self, system_prompt, user_content):
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            **config.GEN_CONFIG
        )
        return response.choices[0].message.content

class Model:
    def __init__(self, client_type="nvidia"):
        # This is where you'll plug in Ollama/VLLM later
        if client_type == "nvidia":
            self.client = NVIDIAClient()
        else:
            raise ValueError(f"Unsupported client: {client_type}")

    def answer(self, context, question):
        # Format the prompt from prompt.py
        formatted_user_prompt = prompt.USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        # Execute calling
        result = self.client.call(
            system_prompt="You are a professional AI assistant.",
            user_content=formatted_user_prompt
        )
        return result

# Example Usage
