# KQI Hindsight Investigator (Lumen)

An autonomous anomaly investigation agent that uses **Hindsight Memory** to form prior hypotheses and verify incident root causes across metric anomalies, deployment events, and configuration changes.

---

## 🌟 Overview

Lumen investigates Key Quality Indicator (KQI) and revenue anomalies by synthesizing historical incident memories with live dataset evidence. By employing Hindsight's `retain`, `recall`, and `reflect` primitives, Lumen continuously learns from analyst feedback and historical incidents while strictly keeping memory as a **prior, not absolute truth**.

### Key Features

* **Memory as a Prior:** Historical incidents inform hypothesis ranking and confidence, but non-memory tool evidence is strictly required to reach a root-cause conclusion.
* **Deterministic Guardrails:** `ToolCallGuard` prevents execution loops, enforces parameter constraints, snaps out-of-window dates, and repairs malformed tool calls.
* **Two-Stage Retain Cycle:** Every investigation retains an unconfirmed report at completion (Stage 1), which is updated when human analyst feedback is submitted (Stage 2).
* **Planted Benchmark Scenarios:** Evaluated against deterministic synthetic scenarios (`SCN-001` through `SCN-004`) covering release regressions, config changes, data quality issues, and traffic anomalies.
* **Interactive Streamlit UI:** Explore active investigations, candidate hypotheses, recalled memory priors, and execution traces in real time.

---

## 📐 Architecture

```
                                  +-----------------------+
                                  |   Streamlit UI / CLI  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  Agent Orchestrator   |
                                  +-----+-----------+-----+
                                        |           |
                     +------------------+           +------------------+
                     |                                                 |
                     v                                                 v
          +--------------------+                            +--------------------+
          |  Memory Service    |                            |  Tool Call Guard   |
          |  (Hindsight SDK)   |                            |  & Data Engine     |
          +---------+----------+                            +---------+----------+
                    |                                                 |
         (retain / recall / reflect)                          (DuckDB SQL Engine)
                    |                                                 |
                    v                                                 v
          +--------------------+                            +--------------------+
          |   Hindsight Cloud  |                            | Metric / Config    |
          |     Memory Bank    |                            | Log Signature Data |
          +--------------------+                            +--------------------+
```

---

## 🚀 Quickstart

### 1. Requirements & Setup

* **Python:** 3.10+
* **Dependencies:** Install required packages using `pip`:

```bash
pip install -r <(echo "
pydantic>=2
pydantic-settings
duckdb
pyarrow
fastparquet
pandas
numpy
groq
hindsight-client
streamlit
uvicorn
pytest
")
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Set the following variables in `.env`:
* `GROQ_API_KEY`: Key for Groq LLM inference.
* `HINDSIGHT_BASE_URL`: Hindsight API endpoint (default: `https://api.hindsight.vectorize.io`).
* `HINDSIGHT_API_KEY`: Your Hindsight API key.
* `HINDSIGHT_BANK`: Memory bank identifier (e.g., `anomaly-investigator`).

---

## 💻 Usage

### Build the Benchmark Database
Generate the DuckDB database populated with planted scenario data:

```bash
python -m data.build --scenarios SCN-001
```

### Run an Investigation
Run an end-to-end investigation CLI on a scenario:

```bash
# Run investigation with Hindsight memory
python -m run_investigation --scenario SCN-001

# Run without memory (baseline comparison)
python -m run_investigation --scenario SCN-001 --no-memory
```

### Seed Memory Bank
Seed historical cases into Hindsight memory:

```bash
python -m scripts.seed_bank --list
```

### Submit Analyst Feedback (Stage 2 Retain)
Submit analyst confirmation or correction for a completed investigation:

```bash
python -m scripts.feedback --list
python -m scripts.feedback <INVESTIGATION_ID> --verdict confirmed --cause "upi-sdk 4.12.0 keepalive regression" --action "rolled back to 4.11.1" --owner platform-eng
```

### Launch Interactive Streamlit App
Start the Streamlit web dashboard:

```bash
streamlit run app/app.py
```

---

## 🧪 Testing & Verification

Run unit tests to verify tool guards, plumbing, and memory prior mechanisms:

```bash
# Run tool execution tests
python -m tests.test_tools

# Run memory prior hypothesis discrimination tests
python -m tests.test_memory_priors

# Run plumbing tests
python -m tests.test_plumbing
```

---

## 📚 Documentation

For deeper details on memory integration, scoring formulas, and design invariants, see:
* [`docs/hindsight_usage.md`](docs/hindsight_usage.md): Detailed guide on Hindsight primitives, prior weighting, and call patterns.
* [`todo.txt`](todo.txt): Roadmap and next implementation steps.
