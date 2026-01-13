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

# Self-Regulated Pipeline Configuration
PIPELINE_CONFIG = {
    "batch_size": 10,
    "max_rules": 15,
    "solver_temperature": 0.0,    # Deterministic for reproducibility
    "optimizer_temperature": 0.3,  # Slight creativity for rule synthesis
    "solver_max_tokens": 1024,
    "optimizer_max_tokens": 2048,
    "early_stop_accuracy": 0.95,
    "early_stop_patience": 3,
}

# Paths
DATASET_PATH = "/root/hsin_research/FinQA-main/dataset/train.json"
CHECKPOINT_DIR = "/root/hsin_research/ruledistill-main/data/checkpoints"
OUTPUT_RULEBOOK_PATH = "/root/hsin_research/ruledistill-main/data/evolved_rulebook.xml"