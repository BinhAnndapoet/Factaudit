# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FACT-AUDIT is an automated framework for evaluating LLM fact-checking capabilities. It uses LangGraph to orchestrate 5 specialized agents that work together to generate test cases, validate evidence, evaluate model responses, and iteratively probe weaknesses in the target LLM.

## Running the System

```bash
# Run the main fact audit evaluation
python src/main.py

# Visualize the graph architecture (exports to img/fact_audit_architecture.png)
python src/visualize_graph.py
```

**Required Setup:**
- Set `GEMINI_API_KEY` in `.env` file (or use Ollama which is pre-configured)
- For Ollama: Ensure `llama3.1` model is pulled and running locally
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

Four specialized LLM instances:
- `llm_explorer` (temp=1.0) - For Appraiser, Prober, Evaluator Phase 1
- `llm_judge` (temp=0.0) - For Inquirer, Quality Inspector, internal judgments
- `llm_scorer` (temp=0.0) - For Evaluator Phase 2
- `llm_target` (temp=0.6) - The model under test

Default uses Ollama with `llama3.1`. To switch to Google Gemini, uncomment the Gemini blocks in config.py.

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
