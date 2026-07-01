# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FACT-AUDIT is an automated framework for evaluating LLM fact-checking capabilities. It uses LangGraph to orchestrate 5 specialized agents that work together to generate test cases, validate evidence, evaluate model responses, and iteratively probe weaknesses in the target LLM.

## Running the System

```bash
# 1. Start the llama-cpp-turboquant servers (dual-model: A + B)
./scripts/start_server.sh baseline      # A-baseline(8080) + B-baseline(8082)
#   --mode turboquant run instead: ./scripts/start_server.sh turboquant
#   Granular / low-VRAM: model-a | model-b | a-turbo | b-turbo | both

# 2. Run the main fact audit evaluation (mode must match the servers you started)
python src/main.py                      # auto mode from .env
python src/main.py --mode baseline      # force Baseline
python src/main.py --mode turboquant    # force TurboQuant+

# Visualize the graph architecture (exports to img/fact_audit_architecture.png)
python src/visualize_graph.py
```

**Required Setup:**
- Two GGUF model files in `models/`: **Model A** `Qwen3-32B-Q8_0.gguf` (5 agents) and **Model B** `Qwen3-14B-Q8_0.gguf` (target). Configure paths/aliases/ports in `.env` (`MODEL_A_*`, `MODEL_B_*`).
- Build the `llama-server` binary from the sibling repo `llama-cpp-turboquant` (default path `../llama-cpp-turboquant/build/bin/llama-server`), or set `LLAMA_SERVER_BIN`.
- Set `TAVILY_API_KEY` in `.env` (required for `web_check_node`). `GEMINI_API_KEY` is optional (cloud fallback only).
- Logs are automatically written to `logs/` with timestamps

## Architecture

### Two-Tier Graph Structure

**Main Graph** - Evolutionary loop that adapts test taxonomy:
1. `inquirer_node` → Generates seed test cases for current task
2. `evaluation_wrapper` → Fan-outs to parallel evaluation subgraphs
3. `aggregate_bad_cases_node` → Collects low-score cases (≤ 3.0)
4. `appraiser_subgraph` → Analyzes weaknesses, proposes new test scenarios

**Evaluation Sub-graph** - Per-case torture test loop:
- `web_check_node` → Validates evidence via Tavily Wikipedia search
- `llm_inspection_node` → LLM-based quality check
- `target_llm_node` → Sends question to target model
- `evaluator_phase1_subgraph` → Generates reference answers
- `evaluator_phase2_score_node` → Scores target response (1-10)
- `prober_node` → Generates harder variants of failed cases (iterative)
- Loops back to web_check until `MAX_ITERATIONS` reached

### The 5 Agents

1. **Appraiser** (`src/appraiser/`) - Analyzes bad cases to propose new test taxonomy categories. High temperature (1.0) for creativity.

2. **Inquirer** (`src/inquirer/`) - Generates initial seed test cases with 3 test modes: `[claim]`, `[evidence]`, `[wisdom of crowds]`. Low temperature (0.0) for consistency.

3. **Quality Inspector** (`src/quality_inspector/`) - Uses Tavily search to verify AI-generated evidence against Wikipedia. Filters hallucinations before evaluation.

4. **Evaluator** (`src/evaluator/`) - Two-phase scoring: Phase 1 generates reference answers via voting, Phase 2 scores target response using LLM-as-a-judge.

5. **Prober** (`src/prober/`) - Performs importance sampling on memory pool (mix of good/bad cases) to generate progressively harder test variants.

### LLM Configuration (src/config.py)

The system uses **two independent models**, each served in two modes (4 `llama-cpp-turboquant` servers total). Mode-switching (Baseline / TurboQuant+) is **global** — one `--mode` flips both models together.

| Role | Model (GGUF) | Baseline | TurboQuant+ | Used by |
|------|--------------|----------|-------------|---------|
| **Model A** | `Qwen3-32B-Q8_0.gguf` | port 8080 (f32) | port 8081 (turbo3/turbo4) | the 5 agents |
| **Model B** | `Qwen3-14B-Q8_0.gguf` | port 8082 (f32) | port 8083 (turbo3/turbo4) | the target model under test |

Four LLM roles map onto these models:
- `llm_explorer` (Model A, temp=1.0) - Appraiser, Prober, Evaluator Phase 1
- `llm_judge` (Model A, temp=0.0) - Inquirer, Quality Inspector, internal judgments
- `llm_scorer` (Model A, temp=0.0) - Evaluator Phase 2
- `llm_target` (Model B, temp=0.6) - the model under test (fully independent of the agents)

`LLMFactory` (singleton via `get_factory`) holds the global mode + two model config dicts (`_model_a` / `_model_b`). Agents read `config.llm_*` at call-time, so `initialize_llms(mode=...)` / `switch_llm_mode(new_mode)` re-point every role transparently. Google Gemini remains available as an optional cloud fallback (`_create_fallback_gemini`) when `GEMINI_API_KEY` is set.

### Key Constants

- `MAX_RETRIES = 3` - LLM rejection retry limit
- `MAX_WEB_CHECKS = 2` - Web verification retry limit
- `LOW_SCORE_THRESHOLD = 3.0` - Threshold for "bad case" classification
- `MAX_ITERATIONS = 3` - Prober loop limit per test case (configurable, paper uses 30)

### State Management

- Uses LangGraph's `Annotated[List[Dict], operator.add]` for concurrent-safe memory pool updates
- `evaluation_wrapper_node` prevents state leakage during parallel fan-out/fan-in
- Main state carries taxonomy scores, aggregated bad cases, and iteration status

### Test Modes

When generating test cases, three modes are used:
1. `[claim]` - Fact verification using only the source claim itself
2. `[evidence]` - Uses Wikipedia-cited evidence (validated via Tavily)
3. `[wisdom of crowds]` - Uses simulated social media conversation trees

## File Structure

```
src/
├── main.py              # Entry point with DualLogger for stdout/stderr capture
├── main_graph.py        # Master graph connecting all agents
├── config.py            # LLM instances and global constants
├── visualize_graph.py   # Export architecture as PNG
├── appraiser/           # Taxonomy analysis agent
├── inquirer/            # Seed test case generation
├── quality_inspector/   # Web verification with Tavily
├── evaluator/           # Two-phase scoring system
├── prober/              # Iterative weakness probing
└── target_model/        # Wrapper for target LLM
```

## Development Notes

- All prompts are structured as LangChain PromptTemplates and defined in `*_prompts.py` files
- Pydantic schemas with `.with_structured_output()` enforce structured LLM responses
- Vietnamese comments in code are from original research implementation
- The system uses map-reduce pattern via LangGraph's `Send` API for parallel evaluation
