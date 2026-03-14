# Ollama Backend Integration Guide

## Overview

Successfully integrated Ollama support into the neuro-DSL pipeline. You can now run the pipeline with local Ollama models (like Qwen 3 Next) instead of the NVIDIA API.

---

## Changes Made

### 1. Enhanced Configuration (`config.py`)

Added Ollama configuration options:

```python
# LLM Backend Selection
LLM_BACKEND = os.getenv("LLM_BACKEND", "nvidia")  # "nvidia" or "ollama"

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-next:latest")
```

### 2. Updated DSLSolverAgent (`dsl_solver_agent.py`)

**New Parameters:**
- `backend`: Choose "nvidia" or "ollama"
- `ollama_model`: Specify Ollama model name

**Initialization:**
```python
# For Ollama
solver = DSLSolverAgent(
    backend='ollama',
    ollama_model='qwen3-next:latest'
)

# For NVIDIA (default)
solver = DSLSolverAgent(backend='nvidia')
```

### 3. Updated OptimizerAgent (`optimizer_agent.py`)

Same backend parameter support as DSLSolverAgent.

### 4. CLI Arguments (`neuro_dsl_pipeline.py`)

**New Arguments:**
```bash
--backend {nvidia,ollama}     # Select LLM backend
--ollama-model MODEL_NAME     # Specify Ollama model
```

---

## Usage Examples

### 1. Using Ollama (Qwen 3 Next on GPU 4)

```bash
# Run with Ollama backend
python neuro_dsl_pipeline.py \
  --backend ollama \
  --ollama-model qwen3-next:latest \
  --train-samples 20 \
  --checkpoint-dir data/checkpoints/ollama_test

# With rule selection enabled
python neuro_dsl_pipeline.py \
  --backend ollama \
  --ollama-model qwen3-next:latest \
  --enable-rule-selection \
  --top-k-rules 5 \
  --train-samples 50
```

### 2. Using NVIDIA API (Llama 3.3 70B)

```bash
# Explicitly specify NVIDIA backend
python neuro_dsl_pipeline.py \
  --backend nvidia \
  --train-samples 20

# Or use default (from config)
python neuro_dsl_pipeline.py --train-samples 20
```

### 3. Using Environment Variables

Set defaults in your environment:

```bash
export LLM_BACKEND=ollama
export OLLAMA_MODEL=qwen3-next:latest
export OLLAMA_BASE_URL=http://localhost:11434/v1

# Now run without --backend flag
python neuro_dsl_pipeline.py --train-samples 20
```

---

## Available Ollama Models

Check your installed models:

```bash
ollama list
```

Current models on your system:
- `qwen3-next:latest` (50 GB) - Currently running on GPU 4
- `gemini-3-pro-preview:latest`

---

## Architecture

### Backend Selection Flow

```
CLIArgs (--backend, --ollama-model)
    ↓
Config defaults (LLM_BACKEND, OLLAMA_MODEL)
    ↓
Agent initialization
    ↓
OpenAI client with appropriate base_url:
  - Ollama: http://localhost:11434/v1
  - NVIDIA: https://integrate.api.nvidia.com/v1
```

### Client Initialization

Both backends use the OpenAI SDK with different endpoints:

**Ollama:**
```python
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Dummy key, Ollama doesn't need auth
)
```

**NVIDIA:**
```python
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)
```

---

## GPU Selection for Ollama

Ollama automatically manages GPU allocation. You can see current usage:

```bash
nvidia-smi
```

Current status:
- **GPU 4**: Running `qwen3-next:latest` (48.9 GB)
- **GPU 0**: Ollama server process (416 MB)

To force Ollama to use specific GPUs:

```bash
# Set CUDA_VISIBLE_DEVICES before starting Ollama
CUDA_VISIBLE_DEVICES=4 ollama serve

# Or in another terminal
CUDA_VISIBLE_DEVICES=4 ollama run qwen3-next
```

---

## Model Performance Comparison

### Expected Differences

| Aspect | NVIDIA Llama 3.3 70B | Ollama Qwen 3 Next |
|--------|----------------------|---------------------|
| Location | Remote API | Local GPU 4 |
| Latency | ~1-2s per request | ~0.5-1s per request |
| Cost | API usage charges | Free (local) |
| Availability | Requires internet | Always available |
| Model Size | 70B parameters | ~50B parameters |

### When to Use Each

**Use NVIDIA API when:**
- Need highest quality responses
- Don't have local GPU availability
- Running large-scale experiments (API handles scaling)

**Use Ollama when:**
- Want to reduce API costs
- Prefer local control
- Need faster iteration cycles
- GPU resources available (GPU 4 is currently free)

---

## Verification Tests

### Test 1: Configuration

```python
from dsl_solver_agent import DSLSolverAgent
import config

print(config.LLM_BACKEND)      # nvidia (default)
print(config.OLLAMA_MODEL)     # qwen3-next:latest
print(config.OLLAMA_BASE_URL)  # http://localhost:11434/v1
```

### Test 2: Agent Initialization

```python
# Ollama backend
solver = DSLSolverAgent(backend='ollama')
print(solver.backend)     # ollama
print(solver.model_name)  # qwen3-next:latest

# NVIDIA backend
solver = DSLSolverAgent(backend='nvidia')
print(solver.backend)     # nvidia
print(solver.model_name)  # meta/llama-3.3-70b-instruct
```

### Test 3: API Connection

```bash
# Test Ollama connection
curl http://localhost:11434/v1/models

# Should return list of available models
```

---

## Troubleshooting

### Issue: "Connection refused" error

**Cause:** Ollama server not running

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# If not running, start it
ollama serve
```

### Issue: Model not found

**Cause:** Model not pulled to local system

**Solution:**
```bash
# Pull the model
ollama pull qwen3-next

# Verify it's available
ollama list
```

### Issue: GPU out of memory

**Cause:** Other processes using GPU 4

**Solution:**
```bash
# Check GPU usage
nvidia-smi

# Use a different GPU or free up memory
# Move to a less-used GPU (e.g., GPU 6 or 7)
CUDA_VISIBLE_DEVICES=6 ollama run qwen3-next
```

### Issue: Empty responses from Ollama

**Cause:** Model may need specific prompt formatting

**Solution:**
- Some models are sensitive to system prompts
- Try adjusting temperature (0.1 to 0.7)
- Check model-specific requirements in Ollama docs

---

## Next Steps

### 1. Test Qwen 3 Next Performance

Run a small test to compare accuracy:

```bash
# Test with Ollama
python neuro_dsl_pipeline.py \
  --backend ollama \
  --train-samples 20 \
  --checkpoint-dir data/checkpoints/qwen_test

# Test with NVIDIA (for comparison)
python neuro_dsl_pipeline.py \
  --backend nvidia \
  --train-samples 20 \
  --checkpoint-dir data/checkpoints/nvidia_test
```

### 2. Optimize for Qwen 3 Next

Different models may need different:
- Temperature settings
- Prompt formatting
- Max tokens limits

### 3. Experiment with Other Ollama Models

```bash
# Try different models
ollama pull llama3.1:70b
ollama pull mixtral:8x7b

# Run pipeline with different model
python neuro_dsl_pipeline.py \
  --backend ollama \
  --ollama-model llama3.1:70b \
  --train-samples 20
```

---

## Summary

✅ **Completed Features:**
1. Multi-backend support (NVIDIA API + Ollama)
2. CLI arguments for backend selection
3. Environment variable configuration
4. Model name specification
5. Backward compatibility with existing NVIDIA workflow

🎯 **Ready to Use:**
- Switch between NVIDIA and Ollama with a single flag
- Run local models on GPU 4 (qwen3-next currently loaded)
- Maintain consistent API across backends (OpenAI SDK)

📊 **Recommended Next Action:**
Test the pipeline with Ollama backend on a small dataset to validate end-to-end functionality and compare performance with NVIDIA backend.
