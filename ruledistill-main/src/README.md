# RuleDistill Source Code

A self-regulating rule augmentation system for improving LLM performance on the FinQA financial question-answering benchmark. The system implements a closed-loop feedback architecture where domain-specific rules dynamically evolve based on solver failures.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Self-Regulated Pipeline                               │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────┐ │
│  │ DataBuffer  │───▶│ SolverAgent  │───▶│     OptimizerAgent          │ │
│  │ (Batches)   │    │ (Reasoning)  │    │ (Rule Evolution)            │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────────┘ │
│                            │                        │                    │
│                            ▼                        ▼                    │
│                    ┌──────────────┐         ┌─────────────┐             │
│                    │  Evaluator   │         │   Rulebook  │◀────────────┘
│                    │  (Metrics)   │         │   (XML)     │
│                    └──────────────┘         └─────────────┘
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧮 Solver Computation Logic

The `SolverAgent` implements Chain-of-Thought (CoT) reasoning with explicit rule application:

### Reasoning Process

1. **Rule Injection**: The rulebook XML is injected into the system prompt as "Law"
2. **Structured Reasoning**: The model outputs reasoning in XML format with explicit rule citations
3. **Answer Extraction**: The final numerical answer is parsed from the `<Answer>` tag
4. **Fallback Parsing**: If XML parsing fails, regex patterns extract the answer

### Answer Cleaning (`_clean_answer`)

```python
# 1. Remove currency symbols and commas
answer = answer.replace("$", "").replace(",", "")

# 2. Remove qualifiers
answer = re.sub(r'^(approximately|about|roughly)\\s*', '', answer)

# 3. Handle percentages - extract number only
if "%" in answer:
    match = re.search(r'(-?\\d+\\.?\\d*)\\s*%', answer)
    return match.group(1)  # Returns "25" not "0.25"

# 4. Extract final number
match = re.search(r'-?\\d+\\.?\\d*', answer)
```

> [!WARNING]
> **Percentage Handling**: The solver extracts the raw number from percentages. A response of "25%" becomes `"25"`, NOT `"0.25"`. This affects downstream comparison logic.

---

## ⚖️ Evaluation Methodology: Old vs New

### Comparison Table

| Aspect | Old Evaluator (`evaluator.py`) | New Evaluator (`finqa_robust_evaluator.py`) |
|--------|-------------------------------|---------------------------------------------|
| **Absolute Tolerance** | `1e-4` (0.0001) | `0.1` |
| **Relative Tolerance** | `1%` (0.01) | `1%` (0.01) |
| **Scale Factor Check** | ❌ None | ✅ 0.001x to 1000x |
| **Ambiguity Resolution** | ❌ None | ✅ Percentage ↔ Decimal |
| **Match Types** | `correct` only | Multiple: `exact_zero`, `absolute_tolerance`, `relative_tolerance`, `scale_*`, `percentage_interpretation_*` |

### Old Evaluator Logic (`evaluator.py`)

```python
def classify_error(pred_val, gt_val, raw_response, tolerance=1e-4, rel_tolerance=0.01):
    # STRICT: abs(gt - pred) < 0.0001 OR relative_error < 1%
    if abs(gt_val - pred_val) < tolerance:
        return "correct"
    if gt_val != 0 and abs((gt_val - pred_val) / gt_val) < rel_tolerance:
        return "correct"
    return "computation error"
```

### New Evaluator Logic (`finqa_robust_evaluator.py`)

```python
class RobustEvaluator:
    def __init__(self, tolerance=0.1, relative_tolerance=0.01,
                 check_scale_factors=True, check_ambiguity=True):
        self.scale_factors = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        
    def evaluate(self, predicted, ground_truth, question):
        # 1. Direct tolerance match (RELAXED: 0.1 absolute, 1% relative)
        if abs(pred - gt) < 0.1:  # ⚠️ 100x more lenient than old method
            return True, "absolute_tolerance"
            
        # 2. Scale invariance check (allows 0.001x to 1000x scaling)
        for scale in self.scale_factors:
            if matches_with_tolerance(pred * scale, gt):
                return True, f"scale_{scale}_match"
                
        # 3. Ambiguity: check if pred=0.25 matches gt=25 (percentage form)
        if is_ambiguous_question(question):  # "how much", "what was the change"
            if matches_with_tolerance(pred * 100, gt):
                return True, "absolute_interpretation_match"
```

### Optimizer Agent Comparison (`optimizer_agent.py`)

The optimizer uses a **10% relative tolerance** for batch evaluation:

```python
def _compare_answers(self, predicted, ground_truth, tolerance=0.10):
    rel_diff = abs(pred_val - gt_val) / abs(gt_val)
    if rel_diff < tolerance:  # 10% relative error allowed
        return True
        
    # Additional: handle percentage ↔ decimal interpretation
    if 80 < scale_ratio < 120 and abs(gt_val) < 1:
        scaled_pred = pred_val / 100  # Try as percentage conversion
```

---

## 📊 Dataset Number Representation Analysis

### Answer Format Distribution (n=6,251)

| Format | Count | Percentage | Examples |
|--------|------:|----------:|----------|
| Percentage symbol (`%`) | 3,508 | 56.1% | `"53%"`, `"-3.2%"` |
| Integer | 1,248 | 20.0% | `"380"`, `"41932"` |
| Decimal number | 1,192 | 19.1% | `"7.4"`, `"56.6"` |
| Million descriptor | 39 | 0.6% | `"$ 13 million"` |
| Thousand descriptor | 7 | 0.1% | `"350824 thousand"` |
| Other | 257 | 4.1% | Empty, text |

### Ground Truth Value Magnitude (`exe_ans`)

| Magnitude Range | Count | Percentage |
|-----------------|------:|----------:|
| = 0 | 20 | 0.3% |
| 0.001 – 0.01 | 88 | 1.4% |
| 0.01 – 0.1 | 836 | 13.6% |
| **0.1 – 0.5** | **1,599** | **26.1%** |
| 0.5 – 1 | 812 | 13.3% |
| 1 – 10 | 745 | 12.2% |
| 10 – 100 | 670 | 10.9% |
| 100 – 1,000 | 514 | 8.4% |
| 1,000 – 10,000 | 337 | 5.5% |
| > 10,000 | 494 | 8.1% |

> [!IMPORTANT]
> **41.4% of all ground truth values are < 0.5 in magnitude.** This is critical for evaluating the tolerance settings.

### Decimal Precision Distribution

| Decimal Places | Count | Percentage | Examples |
|---------------:|------:|----------:|----------|
| 1 | 1,748 | 28.0% | `3.8`, `6.9` |
| 2 | 347 | 5.6% | `0.25`, `-2.44` |
| 3 | 154 | 2.5% | `7.385`, `0.016` |
| 4 | 466 | 7.5% | `0.2023`, `0.1182` |
| **5** | **3,410** | **54.6%** | `41932.20339`, `0.53232` |

---

## ⚠️ Critical Concerns: Tolerance & Precision

### Problem 1: Absolute Tolerance is Too Lenient

The new evaluator uses `tolerance = 0.1`, which creates severe issues for small values:

| Ground Truth | Actual Error Allowed | Example |
|-------------:|---------------------:|---------|
| 0.03 | **333%** | Pred=0.13, GT=0.03 → PASS ❌ |
| 0.05 | **200%** | Pred=0.15, GT=0.05 → PASS ❌ |
| 0.10 | **100%** | Pred=0.20, GT=0.10 → PASS ❌ |
| 0.12 | **83%** | Pred=0.22, GT=0.12 → PASS ❌ |
| 0.25 | **40%** | Pred=0.35, GT=0.25 → PASS ❌ |

> [!CAUTION]
> **With 41.4% of values < 0.5 and 13.6% between 0.01-0.1, the 0.1 absolute tolerance makes the evaluation trivially easy for a large portion of the dataset.**

### Problem 2: Scale Factor Matching is Semantically Wrong

The evaluator accepts answers that differ by factors of 10, 100, or 1000:

```python
scale_factors = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
```

**Example**: If GT = `1500000` (1.5 million), the following are ALL marked correct:
- `1500` → scale factor 1000 ✅
- `15000` → scale factor 100 ✅  
- `150000` → scale factor 10 ✅

> [!CAUTION]
> **In financial domains, "1.5 million" vs "1.5 billion" is a 1000x error that could have catastrophic real-world implications. This tolerance is inappropriate for domain-specific applications.**

### Problem 3: Precision is Under-Specified

**54.6% of ground truths have 5 decimal places** (e.g., `0.53232`), but:

- The old evaluator required `1e-4` (4 decimal places) precision
- The new evaluator accepts `0.1` absolute error — **wrong by 1-2 significant figures**

**Example**:
- GT = `0.03614` (3.614% increase)
- Pred = `0.13` (13% increase)
- New evaluator: ✅ PASS (diff = 0.09386 < 0.1)
- Reality: **260% relative error** — completely wrong answer

### Problem 4: Representation Cannot Be Changed

The FinQA dataset uses specific representations:

| Representation | Meaning | Why It Matters |
|---------------|---------|----------------|
| `0.25` | 25% as decimal | Standard for computations |
| `"25%"` | 25% as percentage | Standard for display |
| `"$ 13 million"` | Currency with scale | Domain convention |

> [!WARNING]
> **Representation is domain-specific and cannot be changed arbitrarily.** An evaluator that accepts `0.25` when the answer should be `25` (or vice versa) is not evaluating the model's understanding—it's hiding errors.

---

## ✅ Recommended Evaluation Settings

For strict, research-grade evaluation:

```python
evaluator = RobustEvaluator(
    tolerance=1e-4,           # Require 4 decimal places
    relative_tolerance=0.01,  # 1% relative error max
    check_scale_factors=False, # DISABLE scale matching
    check_ambiguity=False     # DISABLE ambiguity resolution
)
```

For lenient, exploratory evaluation (acknowledge trade-offs):

```python
evaluator = RobustEvaluator(
    tolerance=0.01,            # 2 decimal places
    relative_tolerance=0.05,   # 5% relative error
    check_scale_factors=False, # Keep disabled
    check_ambiguity=True       # Allow for "how much" questions only
)
```

---

## 📁 Module Reference

### Core Pipeline

| Module | Description |
|--------|-------------|
| [self_regulated_pipeline.py](self_regulated_pipeline.py) | Main orchestrator implementing the closed-loop feedback system with batch processing, checkpointing, and early stopping |
| [solver_agent.py](solver_agent.py) | Chain-of-Thought reasoning agent that uses rulebook as "law" to answer financial questions |
| [optimizer_agent.py](optimizer_agent.py) | Root cause analyzer that synthesizes/refines rules based on solver failures |
| [data_buffer.py](data_buffer.py) | Manages FinQA dataset iteration with batch processing, shuffling, and ground truth extraction |

### Evaluation & Accuracy

| Module | Description |
|--------|-------------|
| [evaluator.py](evaluator.py) | LLM evaluator with rule injection support for testing model accuracy on FinQA |
| [finqa_robust_evaluator.py](finqa_robust_evaluator.py) | Robust numerical comparison with tolerance handling, scale factor detection, and ambiguity resolution |
| [finqa_extractor.py](finqa_extractor.py) | Dataset preprocessor with answer normalization (billions, millions, k) and program execution |
| [judger.py](judger.py) | LLM-based response evaluator comparing model outputs against ground truth |

### Rulebook Management

| Module | Description |
|--------|-------------|
| [rulebook_utils.py](rulebook_utils.py) | XML parsing, serialization, validation, merging, and compression utilities for rulebooks |
| [rule_deduplicator.py](rule_deduplicator.py) | Identifies duplicate/similar rules, clusters by pattern, and generates compressed specialized rulebooks |
| [rule_visualizer.py](rule_visualizer.py) | Generates word clouds, bar charts, network graphs, treemaps, and evolution charts for rulebook analysis |

### Configuration & Utilities

| Module | Description |
|--------|-------------|
| [config.py](config.py) | Central configuration: API keys, model selection, hyperparameters, and file paths |
| [model.py](model.py) | LLM client wrapper for NVIDIA NIM API (OpenAI-compatible) for question answering |
| [prompt.py](prompt.py) | System and user prompt templates for financial reasoning and evaluation tasks |

### `utils/` Subdirectory

| Module | Description |
|--------|-------------|
| `rule.py` | Rule loading and formatting utilities |
| `extract_failures.py` | Extracts failed predictions from evaluation results |
| `inspect_sample.py` | Interactive sample inspection for debugging |
| `compare_results.py` | Compares results between evaluation runs |

---

## 🔄 Pipeline Flow

### 1. Data Loading
The `DataBuffer` loads the FinQA dataset and extracts questions, context, and ground truth answers in configurable batches:

```python
from data_buffer import DataBuffer

buffer = DataBuffer(
    dataset_path="/path/to/train.json",
    batch_size=10,
    shuffle=True
)
```

### 2. Solver Execution
The `SolverAgent` uses Chain-of-Thought reasoning guided by the current rulebook:

```python
from solver_agent import SolverAgent

solver = SolverAgent(client_type="nvidia")
result = solver.predict(
    question="What was the revenue growth?",
    context="Revenue in 2020: $1B, 2021: $1.2B",
    rulebook=current_rulebook_xml
)
```

**Response Format:**
```xml
<Response>
    <Reasoning>Step-by-step calculation...</Reasoning>
    <Answer>0.20</Answer>
    <RulesApplied>01, 03</RulesApplied>
</Response>
```

### 3. Optimization Loop
The `OptimizerAgent` analyzes failures and evolves the rulebook:

```python
from optimizer_agent import OptimizerAgent

optimizer = OptimizerAgent(client_type="nvidia")
result = optimizer.optimize(
    batch_results=solver_results,
    current_rulebook=current_rulebook,
    batch_num=1,
    max_rules=15
)
```

**Error Classification:**
- `MISSING_RULE` → Create new rule
- `BAD_RULE` → Refine rule wording
- `CONFLICTING_RULES` → Merge or delete
- `HALLUCINATION` → Add stricter enforcement

### 4. Full Pipeline
```python
from self_regulated_pipeline import SelfRegulatedPipeline
from data_buffer import DataBuffer
from solver_agent import SolverAgent
from optimizer_agent import OptimizerAgent

pipeline = SelfRegulatedPipeline(
    solver=SolverAgent(),
    optimizer=OptimizerAgent(),
    max_rules=15,
    checkpoint_dir="./checkpoints"
)

results = pipeline.run(
    data_buffer=DataBuffer("train.json", batch_size=10),
    max_batches=50,
    early_stop_accuracy=0.95
)
```

---

## 📊 Rulebook XML Format

```xml
<Rulebook domain="finqa">
    <Rule id="01" type="calculation" phase="generation" confidence="1" source="batch_3">
        <Trigger>When computing percentage change...</Trigger>
        <Action>Use the formula: (new - old) / old × 100</Action>
    </Rule>
    <Rule id="02" type="interpretation" phase="generation" confidence="0.9" source="batch_5">
        <Trigger>When dealing with currency in millions...</Trigger>
        <Action>Convert to full numeric form before calculating</Action>
    </Rule>
</Rulebook>
```

**Rule Attributes:**
- `id`: Unique identifier
- `type`: calculation, interpretation, formatting, validation
- `phase`: generation, verification
- `confidence`: 0.0-1.0 reliability score
- `source`: Batch origin for traceability

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# API Settings
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_NAME = "meta/llama-3.3-70b-instruct"

# Pipeline Configuration
PIPELINE_CONFIG = {
    "batch_size": 10,           # Questions per batch
    "max_rules": 15,            # Maximum rulebook size
    "solver_temperature": 0.0,  # Deterministic solving
    "optimizer_temperature": 0.3,  # Slight creativity
    "early_stop_accuracy": 0.95,   # Target accuracy
    "early_stop_patience": 3,      # Batches without improvement
}

# Paths
DATASET_PATH = "/path/to/FinQA/train.json"
CHECKPOINT_DIR = "/path/to/checkpoints"
```

---

## 📈 Visualization & Analysis

Generate comprehensive rulebook visualizations:

```bash
python rule_visualizer.py \
    --rulebook ./data/evolved_rulebook.xml \
    --checkpoints ./data/checkpoints \
    --output ./analysis
```

**Outputs:**
- `word_cloud.png` - Keyword frequency visualization
- `type_distribution.png` - Rule type breakdown
- `keyword_chart.png` - Top keywords bar chart
- `network_graph.png` - Rule relationship network
- `evolution.png` - Rule count over batches
- `metrics_trend.png` - Accuracy progression
- `rule_summary.html` - Interactive rule table

---

## 🧪 Testing Scripts

| Script | Purpose |
|--------|---------|
| `test_train_set.py` | Evaluate on training split |
| `testset_test.py` | Evaluate on test split |

---

## 🔧 Requirements

- Python 3.10+
- `openai` - API client
- `python-dotenv` - Environment management
- `tqdm` - Progress bars
- `matplotlib`, `wordcloud`, `networkx` - Visualization

---

## 📝 Quick Start

1. **Set up environment:**
   ```bash
   cp .env.example .env
   # Add your NVIDIA_API_KEY to .env
   ```

2. **Run the pipeline:**
   ```bash
   python self_regulated_pipeline.py
   ```

3. **Analyze results:**
   ```bash
   python rule_visualizer.py --rulebook ./data/evolved_rulebook.xml
   ```

---

## 📚 Related Files

- `../data/` - Datasets, checkpoints, and evaluation results
- `../notebook/` - Jupyter notebooks for interactive analysis
- `../output/` - Pipeline execution outputs

---

*This system implements the rule distillation approach for improving LLM numerical reasoning on financial QA tasks through iterative, self-supervised rule synthesis.*
