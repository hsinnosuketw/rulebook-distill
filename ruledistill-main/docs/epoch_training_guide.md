# Epoch Training Feature Example

## 新增功能說明

在 `neuro_dsl_pipeline.py` 中新增了兩個參數，支援在選定的題目數量上進行多個 epoch 的訓練：

### 新增參數

1. **`--train-samples`** (int, optional)
   - 指定用於 epoch 訓練的題目數量
   - 如果不指定，則使用所有題目
   - 系統會自動轉換為 batch 數量（向上取整）

2. **`--epochs`** (int, default=1)
   - 指定在選定題目上訓練的 epoch 數量
   - 預設為 1（單次訓練）

## 使用範例

### 範例 1：在 100 題上訓練 3 個 epochs
```bash
python neuro_dsl_pipeline.py \
  --dataset /path/to/train.json \
  --batch-size 10 \
  --train-samples 100 \
  --epochs 3 \
  --neuro \
  --checkpoint-dir data/checkpoints/epoch_test
```

**輸出結果**：
- 使用前 100 題（10 batches）
- 重複訓練 3 次
- 總共處理 300 個樣本（100 題 × 3 epochs）
- Global batch 編號：0-29（確保 checkpoint 不會互相覆蓋）

### 範例 2：在 50 題上訓練 5 個 epochs
```bash
python neuro_dsl_pipeline.py \
  --batch-size 10 \
  --train-samples 50 \
  --epochs 5 \
  --neuro \
  --sleep-interval 10
```

**效果**：
- 5 batches × 5 epochs = 25 個 global batches
- 每個 epoch 後顯示該 epoch 的準確率摘要
- 適合觀察模型在相同題目上的學習曲線

### 範例 3：使用所有題目訓練 2 個 epochs
```bash
python neuro_dsl_pipeline.py \
  --epochs 2 \
  --max-batches 20
```

**效果**：
- 前 20 個 batches 訓練 2 次
- 不指定 `--train-samples` 時，由 `--max-batches` 控制範圍

## 實現細節

### Global Batch 編號
為了避免 checkpoint 覆蓋，每個 batch 都有唯一的 global batch number：

```
Global Batch # = start_batch + (epoch × effective_batches) + batch_offset
```

**範例**：`--train-samples 30 --batch-size 10 --epochs 3`
- Epoch 1: Global batches 0, 1, 2
- Epoch 2: Global batches 3, 4, 5
- Epoch 3: Global batches 6, 7, 8

### Epoch 統計
多 epoch 訓練時，每個 epoch 結束會顯示：
```
============================================================
EPOCH 2 SUMMARY: 65/100 = 65.0%
============================================================
```

### 與其他參數的互動

- **`--max-batches`**: 指定每個 epoch 的最大 batch 數
- **`--train-samples`**: 會自動計算並覆蓋 `--max-batches`（優先級更高）
- **`--start-batch`**: 仍然有效，從指定的 batch 開始

## 典型使用場景

### 場景 1：測試規則收斂
```bash
# 在 20 題上訓練 10 個 epochs，觀察準確率是否收斂
python neuro_dsl_pipeline.py --train-samples 20 --epochs 10 --batch-size 10
```

### 場景 2：Overfitting 檢測
```bash
# 小數據集多次訓練，檢查是否過擬合
python neuro_dsl_pipeline.py --train-samples 50 --epochs 5
```

### 場景 3：快速迭代實驗
```bash
# 在少量樣本上快速測試新的 sleep-interval 設定
python neuro_dsl_pipeline.py --train-samples 30 --epochs 3 --sleep-interval 5
```

## 注意事項

1. **Checkpoint 管理**：每個 global batch 都會產生獨立的 checkpoint 檔案
2. **記憶體使用**：trace buffer 會累積所有 epochs 的成功樣本
3. **時間成本**：總訓練時間 = 單 epoch 時間 × epochs
4. **Sleep Phase**：仍然按照 `--sleep-interval` 觸發，跨 epochs 計算

## 自訂輸出路徑

### 新增參數

除了基本的 `--checkpoint-dir`，現在支援分別指定不同類型檔案的輸出位置：

1. **`--checkpoint-dir`** (預設: `data/checkpoints/neuro_dsl`)
   - 存放 batch results (`dsl_results_batch_*.jsonl`)
   - 所有其他檔案的預設位置

2. **`--rulebook-dir`** (預設: 與 checkpoint-dir 相同)
   - 存放 rulebook XML 檔案 (`dsl_rulebook_batch_*.xml`)

3. **`--metrics-dir`** (預設: 與 checkpoint-dir 相同)
   - 存放 metrics 日誌 (`dsl_metrics.jsonl`)
   - 存放最終摘要 (`dsl_summary.json`)

### 使用範例

#### 範例 1：所有檔案存在同一目錄
```bash
python neuro_dsl_pipeline.py \
  --checkpoint-dir experiments/exp001
```

**輸出結構**：
```
experiments/exp001/
├── dsl_results_batch_000.jsonl
├── dsl_results_batch_001.jsonl
├── dsl_rulebook_batch_000.xml
├── dsl_rulebook_batch_001.xml
├── dsl_metrics.jsonl
└── dsl_summary.json
```

#### 範例 2：分離 Rulebook 和 Metrics
```bash
python neuro_dsl_pipeline.py \
  --checkpoint-dir data/results \
  --rulebook-dir models/rulebooks \
  --metrics-dir logs/metrics
```

**輸出結構**：
```
data/results/
└── dsl_results_batch_*.jsonl

models/rulebooks/
└── dsl_rulebook_batch_*.xml

logs/metrics/
├── dsl_metrics.jsonl
└── dsl_summary.json
```

#### 範例 3：Epoch 訓練 + 自訂路徑
```bash
python neuro_dsl_pipeline.py \
  --train-samples 100 \
  --epochs 3 \
  --checkpoint-dir experiments/epoch_exp/results \
  --rulebook-dir experiments/epoch_exp/rules \
  --metrics-dir experiments/epoch_exp/logs
```

**優點**：
- 方便版本管理和比較
- 可以重用 rulebook 而不混淆 results
- Metrics 獨立便於分析和可視化

#### 範例 4：只自訂 Metrics 路徑
```bash
python neuro_dsl_pipeline.py \
  --checkpoint-dir data/checkpoints/test1 \
  --metrics-dir centralized_logs/
```

**用途**：
- 將所有實驗的 metrics 集中存放
- 便於跨實驗比較和分析

### 路徑自動創建

所有指定的目錄會自動創建（包含父目錄），無需手動建立。

## 輸出範例

```
============================================================
NEURO-SYMBOLIC DSL PIPELINE
============================================================
Total batches: 626
Batch size: 10
Epochs: 3
Training samples: 50 (reduces to 5 batches)
Checkpoint dir: data/checkpoints/epoch_test
Sleep interval: every 50 batches
============================================================

📚 Epoch Training Mode: 50 samples = 5 batches

============================================================
EPOCH 1/3
============================================================

--- Epoch 1/3, Batch 0/626 (Global #0) ---
...

============================================================
EPOCH 1 SUMMARY: 35/50 = 70.0%
============================================================

============================================================
EPOCH 2/3
============================================================
...
```
