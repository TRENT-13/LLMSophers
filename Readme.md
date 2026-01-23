# LLMSophers: Multi-LLM Collaborative Debate System

A sophisticated multi-agent debate system where four Large Language Models (LLMs) collaboratively solve challenging problems through structured deliberation, peer review, and refinement. The system combats hallucination through diverse perspectives and adversarial review.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Workflow Stages](#workflow-stages)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs and Visualizations](#outputs-and-visualizations)
- [Dataset](#dataset)
- [Technical Details](#technical-details)

## Overview

This project implements a collaborative problem-solving system where:
- **4 LLM Agents** work together to solve complex problems
- **3 Agents** act as independent solvers
- **1 Agent** acts as a final judge
- Solutions undergo **peer review** and **refinement** before final judgment
- The system tracks **performance metrics** and generates **comprehensive visualizations**

The system is designed to improve accuracy by leveraging multiple perspectives and adversarial evaluation, reducing the likelihood of errors that single LLM approaches might produce.

## Architecture

The system follows a **5-stage workflow** implemented using **LangGraph** for state management and orchestration:

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 0: Role Assignment                 │
│  • Each LLM self-assesses preferred role (Solver/Judge)     │
│  • Algorithmic assignment based on confidence scores        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Stage 1: Independent Solution Generation       │
│  • 3 Solvers generate solutions independently               │
│  • No communication between solvers                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Stage 2: Peer Review                     │
│  • Each solver reviews the other two solutions              │
│  • Structured feedback with strengths/weaknesses/errors     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Stage 3: Refinement Based on Feedback          │
│  • Solvers address critiques and refine solutions           │
│  • Explicit acceptance/rejection of feedback                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Stage 4: Final Judgment                  │
│  • Judge evaluates all solutions and selects best           │
│  • Considers original solutions, reviews, and refinements   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
LLMSophers/
├── structure.py              # Core workflow implementation
├── main_runner.py            # Single problem execution script
├── async_runner.py          # Batch processing with metrics
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not in repo)
├── utils/
│   ├── __init__.py
│   └── LLM_call.py         # LLM initialization utilities
└── results/                 # Generated outputs (created at runtime)
    ├── results_*.json       # Full execution results
    ├── summary_*.csv        # Summary statistics
    ├── insights_*.txt       # Generated insights
    └── *.png               # Visualization plots
```

### Core Components

#### `structure.py`
The main module containing the workflow implementation:

- **`AgentState`**: TypedDict defining the state structure passed between nodes
- **`LLM_AGENTS`**: Dictionary mapping agent IDs to initialized LLM instances
- **`judge_election_node()`**: Stage 0 & 0.5 - Role assignment
- **`solver_node()`**: Stage 1 - Parallel solution generation
- **`critic_node()`**: Stage 2 - Peer review phase
- **`refinement_node()`**: Stage 3 - Solution refinement
- **`final_verdict_node()`**: Stage 4 - Final judgment
- **`load_dataset()`**: Loads the problem dataset (65+ challenging problems)

#### `main_runner.py`
Simple execution script for running a single random problem:
- Loads dataset
- Selects random problem
- Executes full workflow
- Prints results

#### `async_runner.py`
Advanced batch processing system:
- Processes entire dataset in configurable batches
- Generates comprehensive metrics and visualizations
- Tracks LLM performance (win rates, accuracy, etc.)
- Saves results in multiple formats (JSON, CSV, pickle)

## Workflow Stages

### Stage 0: Role Assignment

**Self-Assessment Phase:**
- Each of the 4 LLMs receives the problem
- LLMs self-assess their preferred role (Solver or Judge)
- Output format:
  ```json
  {
    "role_preferences": ["Solver", "Judge"],
    "confidence_by_role": {
      "Solver": 0.85,
      "Judge": 0.75
    },
    "reasoning": "..."
  }
  ```

**Algorithmic Assignment:**
- Deterministic algorithm selects judge based on:
  1. Highest judge confidence score
  2. Preference for judge role (tiebreaker)
- Remaining 3 agents become solvers

### Stage 1: Independent Solution Generation

- All 3 solvers work **independently** and **in parallel** (async)
- No communication between solvers
- Each generates complete solution with:
  - Approach and reasoning
  - Step-by-step calculations
  - Final answer

### Stage 2: Peer Review Round

- Each solver reviews the **other two** solutions
- Structured feedback format:
  ```json
  {
    "strengths": [...],
    "weaknesses": [...],
    "errors": [
      {
        "location": "Step X",
        "error_type": "logical_error",
        "description": "...",
        "severity": "critical/minor"
      }
    ],
    "suggested_changes": [...],
    "overall_score": 0-10
  }
  ```
- All reviews run **in parallel** (async)

### Stage 3: Refinement Based on Feedback

- Each solver receives 2 peer reviews
- Must explicitly address each critique:
  - Accept and fix
  - Reject with justification
- Output format:
  ```json
  {
    "changes_made": [
      {
        "critique": "...",
        "response": "...",
        "accepted": true/false
      }
    ],
    "refined_solution": "...",
    "refined_answer": "...",
    "confidence": 0.0-1.0
  }
  ```

### Stage 4: Final Judgment

- Judge receives:
  - All 3 original solutions
  - All peer reviews
  - All 3 refined solutions
- Judge evaluates and selects best solution
- Output format:
  ```json
  {
    "winner": "Agent Name",
    "confidence": 0.0-1.0,
    "reasoning": "...",
    "final_answer_text": "..."
  }
  ```

## Installation

### Prerequisites

- Python 3.8+
- API keys for LLM providers (OpenAI and/or DeepSeek)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TRENT-13/LLMSophers
   cd LLMSophers
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   DEEPSEEK_API_KEY=your_deepseek_key_here
   ```

4. **Verify LLM configuration:**
   Edit `structure.py` to configure your LLM agents in the `LLM_AGENTS` dictionary (lines ~873-878).

## Configuration

### LLM Agents

The system uses 4 LLM agents defined in `structure.py`:

```python
LLM_AGENTS = {
    "GPT-Creative": gpt1,           # GPT-4o-mini, temperature=0.9
    "GPT-Analytical": gpt2,         # GPT-4o-mini, temperature=0.2
    "DeepSeek-Balanced": deepseek1, # DeepSeek Chat, temperature=0.7
    "DeepSeek-Strict": deepseek2    # DeepSeek Chat, temperature=0.0
}
```

You can modify these to use different models or configurations.

### Async Runner Configuration

In `async_runner.py`, adjust `max_concurrent` (line ~646) based on your API rate limits:
```python
max_concurrent = 20  # Number of problems processed simultaneously per batch
```

## Usage

### Running a Single Problem

Execute a random problem from the dataset:

```bash
python main_runner.py
```

This will:
- Load the dataset
- Select a random problem
- Execute the full 5-stage workflow
- Print results to console

### Running the Full Dataset

Process all problems with metrics and visualizations:

```bash
python async_runner.py
```

This will:
- Process all problems in batches
- Generate comprehensive metrics
- Create visualizations
- Save results to `results/` directory

**Expected runtime:** Depends on API latency and batch size. With 65 problems and `max_concurrent=20`, approximately 10-30 minutes.

## Outputs and Visualizations

### Results Directory Structure

After running `async_runner.py`, the `results/` directory contains:

#### Data Files
- **`results_YYYYMMDD_HHMMSS.json`**: Complete execution data for all problems
- **`summary_YYYYMMDD_HHMMSS.csv`**: Summary statistics (CSV format)
- **`summary_YYYYMMDD_HHMMSS.pkl`**: Summary DataFrame (pickle format)
- **`insights_YYYYMMDD_HHMMSS.txt`**: Generated insights and metrics

#### Visualizations

The system generates **12 visualization plots**:

1. **`success_rate_*.png`**: Problem execution success rate (pie chart)
2. **`execution_time_dist_*.png`**: Distribution of execution times (histogram)
3. **`judge_selection_*.png`**: Judge selection frequency (bar chart)
4. **`problem_type_dist_*.png`**: Problem type distribution (bar chart)
5. **`judge_by_type_heatmap_*.png`**: Judge selection by problem type (heatmap)
6. **`exec_time_by_type_*.png`**: Execution time by problem type (box plot)
7. **`llm_win_rates_*.png`**: LLM win rates as solver (bar chart)
8. **`llm_accuracy_rates_*.png`**: LLM accuracy rates (bar chart)
9. **`llm_judge_accuracy_*.png`**: LLM accuracy as judge (bar chart)
10. **`llm_participation_*.png`**: LLM participation (solver vs judge, grouped bars)
11. **`overall_accuracy_*.png`**: Overall system accuracy (pie chart)
12. **`llm_performance_comparison_*.png`**: Win rate vs accuracy comparison (grouped bars)

### Key Metrics Tracked

- **System-Level:**
  - Overall accuracy (correct/incorrect solutions)
  - Success rate (completed/failed problems)
  - Execution time statistics

- **LLM-Level:**
  - Win rates (percentage of solutions selected as winner)
  - Accuracy rates (percentage of correct solutions)
  - Judge accuracy (percentage of correct selections when judging)
  - Participation counts (times as solver vs judge)

- **Problem-Level:**
  - Distribution by type (math, physics, logic, game theory, etc.)
  - Performance by problem type
  - Judge selection patterns

## Dataset

The dataset (`structure.py`, lines 30-824) contains **65+ challenging problems** across multiple categories:

- **Mathematics & Number Theory** (15 problems)
  - Infinite power towers, Borwein integrals, Hilbert matrices, etc.

- **Physics & Engineering** (18 problems)
  - Relativistic paradoxes, quantum effects, Maxwell's demon, etc.

- **Logic & Computer Science** (15 problems)
  - Knights and knaves, blue-eyed islanders, quines, etc.

- **Game Theory & Strategy** (17 problems)
  - Dollar auctions, prisoner's dilemmas, mechanism design, etc.

Each problem includes:
- `field`: Category (STEM)
- `type`: Subcategory (math, physics, logic, game_theory, etc.)
- `question`: Problem statement
- `answer`: Expected correct answer

## Technical Details

### State Management

The system uses **LangGraph** for workflow orchestration. State flows through nodes via the `AgentState` TypedDict:

```python
class AgentState(TypedDict):
    question: str
    question_type: str
    field: str
    agent_names: dict
    elected_judge_name: str
    elected_judge_id: str
    solver_ids: list[str]
    solver_names: list[str]
    solver_answers: list[dict]
    critic_feedback: list[str]
    refined_answer: str
    final_verdict: str
    confidence_score: float
    messages: Annotated[list, operator.add]
```

### Concurrency

- **Solver solutions**: Run in parallel using `asyncio.gather()`
- **Peer reviews**: All 6 reviews (3 solvers × 2 reviews each) run concurrently
- **Batch processing**: Async runner processes problems in configurable batches

### Agent Name Disambiguation

The system ensures unique agent names by:
- Tracking name usage across problems
- Adding numeric suffixes for duplicates (e.g., "Augustus_1", "Augustus_2")
- Maintaining consistency within each problem execution

### Error Handling

- Failed problems are logged with error messages and tracebacks
- Metrics distinguish between successful and failed executions
- Results are saved even if visualization generation fails

## Authors

GBL Team:
Giorgi Kochlamazashvili
Terenti Kaxniashvili
