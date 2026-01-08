import os
from dotenv import load_dotenv
load_dotenv()
# API Settings
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Model Selection
MODEL_NAME = "meta/llama-3.3-70b-instruct"

# Hyper-parameters
GEN_CONFIG = {
    "temperature": 0.1,  # Low temperature is standard for "Judger" tasks
    "top_p": 0.7,
    "max_tokens": 1024,
    "stream": False
}