``` mermaid
flowchart TD
    %% Define styles
    classDef system fill:#f9f2f4,stroke:#333,stroke-width:2px;
    classDef memory fill:#e1f5fe,stroke:#c2185b,stroke-width:2px;
    classDef agent fill:#e8f5e9,stroke:#0277bd,stroke-width:2px;
    classDef decision fill:#fff8e1,stroke:#fbc02d,stroke-width:2px;
    classDef dataset fill:#f3e5f5,stroke:#388e3c,stroke-width:2px;

    %% Components
    Dataset[("Domain Dataset
(FinQA/Legal/Med)")]:::dataset

    subgraph Memory["Hierarchical Memory System"]
        Rulebook[("Global Index
(rulebook.md)")]:::memory
        Examples[("Low-level Examples
(/rules_example/*.md)")]:::memory
        Rulebook -. "Points to" .-> Examples
    end

    subgraph Inference["Inference Phase (Single-turn QA)"]
        Solver["Solver Agent"]:::agent
        Retriever["Context-Aware Retriever"]:::system
        CoT["Chain-of-Thought
(Reasoning & Answer)"]:::system
    end

    subgraph Evaluation["Evaluation & Buffering"]
        Evaluator{"Answer Correct?"}:::decision
        Buffer[("Error Buffer
(Wait until N=5 errors)")]:::memory
    end

    subgraph Evolution["Self-Evolution Phase (Batch Update)"]
        Optimizer["Rulebook Optimizer Agent"]:::agent
        Analyzer["Analyze Error Trajectory
(Why did CoT fail?)"]:::system
        UpdateLogic{"Action Type"}:::decision
        UpdateTriggers["Update Keywords
(Rulebook.md)"]:::system
        AddEdgeCase["Append Edge Case
(Example.md)"]:::system
        CreateNew["Create New Rule
(Both)"]:::system
    end

    %% Flow - Inference
    Dataset -->|"1. New Question"| Retriever
    Rulebook -->|"2. Match Keywords"| Retriever
    Examples -->|"3. Retrieve relevant md"| Retriever
    Retriever -->|"4. Prompt: Q + Retrieved Rules"| Solver
    Solver -->|"5. Generate"| CoT

    %% Flow - Evaluation
    CoT -->|"6. Output"| Evaluator
    Evaluator -->|"Correct (Success)"| NextQ["Next Question"]
    Evaluator -->|"Wrong (Failure)"| Buffer

    %% Flow - Evolution
    Buffer -->|"7. Batch of 5 errors
(Q + Wrong CoT + Target)"| Analyzer
    Analyzer -->|"8. Abstract to Rule"| Optimizer
    Optimizer --> UpdateLogic
    UpdateLogic -->|"Rule unused"| UpdateTriggers
    UpdateLogic -->|"Rule misinterpreted"| AddEdgeCase
    UpdateLogic -->|"No rule exists"| CreateNew

    UpdateTriggers -.-> Rulebook
    AddEdgeCase -.-> Examples
    CreateNew -.-> Rulebook
    CreateNew -.-> Examples
```

系統實作設計（AI Workflow），你可以直接將這個架構對應到 LangChain、LangGraph 或 AutoGen 的程式碼實作中。這個架構精準針對「單輪 QA（Single-turn QA）」設計，並實作了我們先前討論的「高階抽象/低層範例更新機制」。

## 核心系統架構 (System Architecture)

系統分為三個主要階段：**Inference (推論)**、**Evaluation (評估與緩衝)**、以及 **Evolution (自我演化)**。

### 階段一：Inference (推論) - Solver Agent
這是系統回答單輪 QA 的當下階段。

1. **New Question Arrives**: 系統接收到新的問題（例如 FinQA 題：計算 A 公司 2020 年與 2021 年淨利的年增率）。
2. **Context-Aware Retrieval**: 
   - 檢索器（Retriever）先讀取輕量級的全局目錄 `rulebook.md`，比對題目關鍵字（如「年增率」、「淨利」）。
   - 根據目錄的指示，從 `/rules_example/` 資料夾中提取出 1-2 個最相關的低階錯誤範例檔案。
3. **Prompt Construction**: 將問題、相關的外部文件（如財報段落），以及「檢索到的規則與防錯範例」組合進 System Prompt。
4. **CoT Generation**: Solver Agent 被要求嚴格輸出三個段落：
   - `<retrieval_plan>`：確認自己要找什麼資料。
   - `<reasoning_steps>`：運算邏輯與步驟。
   - `<final_answer>`：最終答案。

### 階段二：Evaluation (評估與緩衝)
這個階段負責捕捉錯誤軌跡，但不立即更新，避免規則震盪。

1. **Answer Evaluation**: 由於你使用的是有標準答案的 Benchmark（如 FinQA），系統會自動比對 `<final_answer>` 與 Ground Truth。
   - 如果正確：進入下一題。
   - 如果錯誤：將 `[問題, 外部文件, Solver的 CoT 軌跡, 當時給的 Rules, 正確答案]` 打包存入 **Error Buffer (錯誤緩衝區)**。
2. **Batch Trigger**: 當 Error Buffer 累積滿 5 題錯題時，觸發階段三。

### 階段三：Evolution (自我演化) - Optimizer Agent
這是你的核心創新點，不改變模型權重，而是 CRUD 修改記憶結構。

1. **Trajectory Analysis (軌跡分析)**：Optimizer 讀取這 5 題的 CoT 軌跡，找出 Solver 到底哪裡想錯了。
2. **Action Type Decision (動作決策)**：針對每個錯誤，Optimizer 必須呼叫三個預設 Tool 之一：
   - **Update Keywords (修正觸發條件)**：如果錯誤是因為 Solver 根本沒拿到對的 Rule。Optimizer 呼叫工具在 `rulebook.md` 中現有規則的 `[Trigger]` 加上新的關鍵字。
   - **Append Edge Case (補充邊界條件)**：如果 Solver 拿了正確的 Rule 卻誤解（例如套錯公式）。Optimizer 呼叫工具，在對應的 `/rules_example/xxx.md` 底下新增一個負面教材（Negative Example）。
   - **Create New Rule (創建新規則)**：如果現有規則都無法涵蓋這個錯誤。Optimizer 呼叫工具生成全新的高階描述（寫入 `rulebook.md`）並建立全新的範例檔（寫入 `/rules_example/`）。

***

## 程式實作建議 (Tech Stack)

要在這 3.5 個月內實作出來並趕上 6 月底的口試，我建議你使用以下工具鏈，會比從頭刻 Python 快非常多：

- **Agent 框架**: 使用 **LangGraph**。因為你的系統有明確的「狀態流轉（State Routing）」，例如累積滿 5 題就從 Inference 狀態轉移到 Evolution 狀態，LangGraph 的狀態機設計非常適合這種 Batch 處理。
- **Tool Calling**: 依賴 GPT-4o 或 Claude 3.5 Sonnet 的 `bind_tools` 能力。只要定義好 Pydantic Schema（如 `class UpdateRule(BaseModel): rule_id, new_keywords`），模型就能精準輸出你要的更新指令。
- **Memory 儲存**: 初期不需要用向量資料庫（Vector DB）。既然是維護 Markdown，直接使用 Python 的 `os` 和 `json` 進行本地檔案的讀寫（Local File CRUD）即可，這樣在做 Ablation Study 檢視 `rulebook.md` 的演化過程時會非常直觀。

