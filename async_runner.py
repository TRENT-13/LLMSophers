import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from structure import (
    load_dataset,
    judge_election_node,
    solver_node,
    critic_node,
    refinement_node,
    final_verdict_node,
    AgentState,
    LLM_AGENTS
)


# Configure plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def ensure_unique_names(agent_names: Dict[str, str]) -> Dict[str, str]:
    """
    Ensure all agent names are unique by adding disambiguation suffixes if needed.
    
    Args:
        agent_names: Dictionary mapping agent_id to name
        
    Returns:
        Dictionary with unique names (may have suffixes like "Name_1", "Name_2")
    """
    name_counts = Counter(agent_names.values())
    unique_names = {}
    name_usage = defaultdict(int)
    
    # Process in sorted order of agent_id for determinism
    for agent_id in sorted(agent_names.keys()):
        name = agent_names[agent_id]
        if name_counts[name] > 1:
            # Name is duplicated, add suffix (first occurrence keeps original name)
            name_usage[name] += 1
            if name_usage[name] == 1:
                # First occurrence keeps original name
                unique_names[agent_id] = name
            else:
                # Subsequent occurrences get suffix
                unique_name = f"{name}_{name_usage[name] - 1}"
                unique_names[agent_id] = unique_name
        else:
            unique_names[agent_id] = name
    
    return unique_names


def run_single_problem(problem_idx: int, problem_data: pd.Series) -> Dict[str, Any]:
    """
    Run a single problem through the full workflow.
    
    Args:
        problem_idx: Index of the problem in the dataset
        problem_data: Series containing question, answer, field, type
        
    Returns:
        Dictionary containing all results and metadata
    """
    start_time = time.time()
    
    # Initialize state
    state: AgentState = {
        "question": problem_data['question'],
        "question_type": "",
        "field": "",
        "agent_names": {},
        "election_responses": [],
        "judge_votes": {},
        "elected_judge_name": "",
        "elected_judge_id": "",
        "solver_ids": [],
        "solver_names": [],
        "solver_answers": [],
        "best_answer": "",
        "critic_feedback": [],
        "refined_answer": "",
        "final_verdict": "",
        "confidence_score": 0.0,
        "messages": []
    }
    
    result = {
        "problem_idx": problem_idx,
        "field": problem_data['field'],
        "type": problem_data['type'],
        "question": problem_data['question'],
        "expected_answer": problem_data['answer'],
        "success": False,
        "error": None,
        "execution_time": 0.0,
        "timestamp": datetime.now().isoformat(),
        "state": None
    }
    
    try:
        # Run all nodes
        state = judge_election_node(state)
        
        # Ensure unique names
        original_names = state["agent_names"].copy()
        state["agent_names"] = ensure_unique_names(state["agent_names"])
        
        # Update references if names changed - create mapping by agent_id
        if original_names != state["agent_names"]:
            # Build mapping: original_name -> new_name for each agent_id
            name_mapping = {}
            for agent_id in original_names.keys():
                orig_name = original_names[agent_id]
                new_name = state["agent_names"][agent_id]
                if orig_name != new_name:
                    name_mapping[orig_name] = new_name
            
            # Update elected judge name
            if state["elected_judge_name"] in name_mapping:
                state["elected_judge_name"] = name_mapping[state["elected_judge_name"]]
            
            # Update solver names
            state["solver_names"] = [name_mapping.get(name, name) for name in state["solver_names"]]
            
            # Update election responses
            for resp in state["election_responses"]:
                if resp["agent_name"] in name_mapping:
                    resp["agent_name"] = name_mapping[resp["agent_name"]]
        
        state = solver_node(state)
        state = critic_node(state)
        state = refinement_node(state)
        state = final_verdict_node(state)
        
        result["success"] = True
        result["state"] = {
            "agent_names": state["agent_names"],
            "elected_judge_name": state["elected_judge_name"],
            "elected_judge_id": state["elected_judge_id"],
            "solver_names": state["solver_names"],
            "solver_ids": state["solver_ids"],
            "judge_votes": state["judge_votes"],
            "solver_answers": [
                {
                    "agent_id": sol["agent_id"],
                    "agent_name": sol["agent_name"],
                    "answer_length": len(sol["answer"]),
                    "answer_preview": sol["answer"][:200] + "..." if len(sol["answer"]) > 200 else sol["answer"]
                }
                for sol in state["solver_answers"]
            ],
            "critic_feedback_count": len(state["critic_feedback"]),
            "final_verdict": state["final_verdict"],
            "question_type": state["question_type"]
        }
        
    except Exception as e:
        result["error"] = str(e)
        result["error_traceback"] = traceback.format_exc()
    
    result["execution_time"] = time.time() - start_time
    return result


async def run_problem_async(problem_idx: int, problem_data: pd.Series, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """
    Async wrapper for running a single problem.
    
    Args:
        problem_idx: Index of the problem
        problem_data: Problem data
        semaphore: Semaphore to limit concurrent executions
        
    Returns:
        Result dictionary
    """
    async with semaphore:
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_single_problem, problem_idx, problem_data)
        return result


async def run_all_problems_async(max_concurrent: int = 2) -> List[Dict[str, Any]]:
    """
    Run all problems asynchronously in batches.
    
    Args:
        max_concurrent: Maximum number of concurrent problem executions per batch
        
    Returns:
        List of all results
    """
    df = load_dataset()
    total_problems = len(df)
    print(f"\n{'='*80}")
    print(f"Starting async execution of {total_problems} problems")
    print(f"Max concurrent per batch: {max_concurrent}")
    print(f"Total batches: {(total_problems + max_concurrent - 1) // max_concurrent}")
    print(f"{'='*80}\n")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    
    # Process in batches
    problem_list = [(idx, row) for idx, row in df.iterrows()]
    
    for batch_start in range(0, total_problems, max_concurrent):
        batch_end = min(batch_start + max_concurrent, total_problems)
        batch = problem_list[batch_start:batch_end]
        batch_num = (batch_start // max_concurrent) + 1
        total_batches = (total_problems + max_concurrent - 1) // max_concurrent
        
        print(f"Processing batch {batch_num}/{total_batches} (problems {batch_start+1}-{batch_end})...")
        
        # Create tasks for this batch
        tasks = [
            run_problem_async(idx, row, semaphore)
            for idx, row in batch
        ]
        
        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        print(f"  ✓ Batch {batch_num} completed ({len(batch_results)} problems)\n")
    
    # Sort by problem_idx to maintain order
    results.sort(key=lambda x: x["problem_idx"])
    
    return results


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """
    Save results to JSON and pickle files.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results as JSON
    json_path = output_dir / f"results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved full results to {json_path}")
    
    # Save summary DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            "problem_idx": r["problem_idx"],
            "field": r["field"],
            "type": r["type"],
            "success": r["success"],
            "execution_time": r["execution_time"],
            "error": r.get("error"),
            "judge_name": r.get("state", {}).get("elected_judge_name", "N/A"),
            "judge_id": r.get("state", {}).get("elected_judge_id", "N/A"),
            "num_solvers": len(r.get("state", {}).get("solver_names", [])),
            "num_critics": r.get("state", {}).get("critic_feedback_count", 0),
            "question_type": r.get("state", {}).get("question_type", "N/A")
        })
    
    df_summary = pd.DataFrame(summary_data)
    pickle_path = output_dir / f"summary_{timestamp}.pkl"
    df_summary.to_pickle(pickle_path)
    print(f"Saved summary DataFrame to {pickle_path}")
    
    csv_path = output_dir / f"summary_{timestamp}.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")
    
    return df_summary, json_path, pickle_path


def extract_winner_from_verdict(verdict_text: str, solver_names: List[str]) -> Optional[str]:
    """
    Extract winner from final verdict JSON.
    """
    import re
    try:
        # Try to parse JSON
        if "```json" in verdict_text:
            verdict_text = verdict_text.split("```json")[1].split("```")[0].strip()
        elif "```" in verdict_text:
            verdict_text = verdict_text.split("```")[1].split("```")[0].strip()
        
        verdict_json = json.loads(verdict_text)
        winner = verdict_json.get("winner", "")
        
        # Match winner to solver name
        for solver_name in solver_names:
            if solver_name.lower() in winner.lower() or winner.lower() in solver_name.lower():
                return solver_name
        return winner if winner else None
    except:
        # Fallback: try to find solver name in text
        for solver_name in solver_names:
            if solver_name.lower() in verdict_text.lower():
                return solver_name
        return None


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics from results, including LLM performance.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic statistics
    total_problems = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total_problems - successful
    
    metrics["total_problems"] = total_problems
    metrics["successful"] = successful
    metrics["failed"] = failed
    metrics["success_rate"] = successful / total_problems if total_problems > 0 else 0
    
    # Execution time statistics
    exec_times = [r["execution_time"] for r in results if r["success"]]
    if exec_times:
        metrics["avg_execution_time"] = np.mean(exec_times)
        metrics["median_execution_time"] = np.median(exec_times)
        metrics["min_execution_time"] = np.min(exec_times)
        metrics["max_execution_time"] = np.max(exec_times)
        metrics["std_execution_time"] = np.std(exec_times)
    
    # Judge selection statistics
    judge_selections = Counter()
    judge_by_type = defaultdict(Counter)
    judge_by_field = defaultdict(Counter)
    
    # LLM Performance Tracking
    llm_as_solver_count = Counter()  # How many times each LLM was a solver
    llm_as_judge_count = Counter()   # How many times each LLM was judge
    llm_wins = Counter()             # How many times each LLM's solution won
    llm_solver_participations = Counter()  # Total times as solver
    
    # Accuracy tracking
    problems_correct = 0
    problems_incorrect = 0
    
    # Per-LLM accuracy
    llm_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    llm_judge_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # Before/after refinement tracking
    solver_performance = defaultdict(lambda: {"wins": 0, "participations": 0})
    
    for r in results:
        if r["success"] and r.get("state"):
            state = r["state"]
            judge_id = state.get("elected_judge_id", "Unknown")
            judge_name = state.get("elected_judge_name", "Unknown")
            problem_type = r["type"]
            field = r["field"]
            expected_answer = r.get("expected_answer", "").lower()
            final_verdict = state.get("final_verdict", "")
            solver_answers = state.get("solver_answers", [])
            solver_ids = state.get("solver_ids", [])
            solver_names = state.get("solver_names", [])
            
            # Judge statistics
            judge_selections[judge_id] += 1
            judge_by_type[problem_type][judge_id] += 1
            judge_by_field[field][judge_id] += 1
            llm_as_judge_count[judge_id] += 1
            
            # Solver statistics
            for solver_id in solver_ids:
                llm_as_solver_count[solver_id] += 1
                llm_solver_participations[solver_id] += 1
                solver_performance[solver_id]["participations"] += 1
            
            # Extract winner from verdict
            winner_name = extract_winner_from_verdict(final_verdict, solver_names)
            if winner_name and solver_answers:
                # Find winner's agent_id
                for sol in solver_answers:
                    if sol.get("agent_name") == winner_name:
                        winner_id = sol.get("agent_id")
                        llm_wins[winner_id] += 1
                        solver_performance[winner_id]["wins"] += 1
                        break
            
            # Check if final answer is correct (simple keyword matching)
            verdict_lower = final_verdict.lower()
            is_correct = False
            if expected_answer:
                # Extract key numbers/answers from expected answer
                expected_keywords = set(re.findall(r'\b\d+\.?\d*\b', expected_answer))
                verdict_keywords = set(re.findall(r'\b\d+\.?\d*\b', verdict_lower))
                
                # Check for common keywords
                common_words = set(expected_answer.split()[:10])  # First 10 words
                if any(word in verdict_lower for word in common_words if len(word) > 3):
                    is_correct = True
                elif expected_keywords and verdict_keywords:
                    # Check if any key numbers match
                    if expected_keywords.intersection(verdict_keywords):
                        is_correct = True
            
            if is_correct:
                problems_correct += 1
                # Award accuracy to winner
                if winner_name and solver_answers:
                    for sol in solver_answers:
                        if sol.get("agent_name") == winner_name:
                            winner_id = sol.get("agent_id")
                            llm_accuracy[winner_id]["correct"] += 1
                            llm_accuracy[winner_id]["total"] += 1
                            llm_judge_accuracy[judge_id]["correct"] += 1
                            llm_judge_accuracy[judge_id]["total"] += 1
                            break
            else:
                problems_incorrect += 1
                # Still count total
                if winner_name and solver_answers:
                    for sol in solver_answers:
                        if sol.get("agent_name") == winner_name:
                            winner_id = sol.get("agent_id")
                            llm_accuracy[winner_id]["total"] += 1
                            llm_judge_accuracy[judge_id]["total"] += 1
                            break
    
    metrics["judge_selections"] = dict(judge_selections)
    metrics["judge_by_type"] = {k: dict(v) for k, v in judge_by_type.items()}
    metrics["judge_by_field"] = {k: dict(v) for k, v in judge_by_field.items()}
    
    # LLM Performance Metrics
    metrics["llm_as_solver_count"] = dict(llm_as_solver_count)
    metrics["llm_as_judge_count"] = dict(llm_as_judge_count)
    metrics["llm_wins"] = dict(llm_wins)
    metrics["llm_solver_participations"] = dict(llm_solver_participations)
    
    # Calculate win rates
    metrics["llm_win_rates"] = {
        llm_id: wins / llm_solver_participations[llm_id] 
        if llm_solver_participations[llm_id] > 0 else 0
        for llm_id, wins in llm_wins.items()
    }
    
    # Calculate accuracy rates
    metrics["llm_accuracy_rates"] = {
        llm_id: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for llm_id, stats in llm_accuracy.items()
    }
    
    metrics["llm_judge_accuracy_rates"] = {
        llm_id: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for llm_id, stats in llm_judge_accuracy.items()
    }
    
    # Store raw accuracy data
    metrics["llm_accuracy"] = {k: dict(v) for k, v in llm_accuracy.items()}
    metrics["llm_judge_accuracy"] = {k: dict(v) for k, v in llm_judge_accuracy.items()}
    
    # Overall accuracy
    metrics["overall_accuracy"] = problems_correct / successful if successful > 0 else 0
    metrics["problems_correct"] = problems_correct
    metrics["problems_incorrect"] = problems_incorrect
    
    # Problem type distribution
    type_distribution = Counter(r["type"] for r in results)
    metrics["type_distribution"] = dict(type_distribution)
    
    # Field distribution
    field_distribution = Counter(r["field"] for r in results)
    metrics["field_distribution"] = dict(field_distribution)
    
    # Error analysis
    errors = [r["error"] for r in results if r.get("error")]
    error_types = Counter(str(e)[:50] if e else "Unknown" for e in errors)
    metrics["error_types"] = dict(error_types)
    
    # Agent name uniqueness
    all_names = []
    for r in results:
        if r["success"] and r.get("state"):
            names = r["state"].get("agent_names", {})
            all_names.extend(names.values())
    
    name_counts = Counter(all_names)
    duplicate_names = {name: count for name, count in name_counts.items() if count > 1}
    metrics["duplicate_names"] = duplicate_names
    metrics["unique_name_rate"] = len(set(all_names)) / len(all_names) if all_names else 1.0
    
    # Solver answer length statistics
    answer_lengths = []
    for r in results:
        if r["success"] and r.get("state"):
            solver_answers = r["state"].get("solver_answers", [])
            for sol in solver_answers:
                answer_lengths.append(sol.get("answer_length", 0))
    
    if answer_lengths:
        metrics["avg_answer_length"] = np.mean(answer_lengths)
        metrics["median_answer_length"] = np.median(answer_lengths)
        metrics["min_answer_length"] = np.min(answer_lengths)
        metrics["max_answer_length"] = np.max(answer_lengths)
    
    return metrics


def generate_insights(metrics: Dict[str, Any], results: List[Dict[str, Any]]) -> List[str]:
    """
    Generate comprehensive insights from metrics and results.
    
    Args:
        metrics: Calculated metrics
        results: All results
        
    Returns:
        List of insight strings
    """
    insights = []
    
    # Success rate insight
    success_rate = metrics["success_rate"]
    if success_rate >= 0.95:
        insights.append(f"✓ Excellent success rate: {success_rate:.1%} of problems completed successfully")
    elif success_rate >= 0.80:
        insights.append(f"⚠ Good success rate: {success_rate:.1%}, but {metrics['failed']} problems failed")
    else:
        insights.append(f"✗ Low success rate: {success_rate:.1%} - {metrics['failed']} problems failed")
    
    # Overall accuracy
    overall_accuracy = metrics.get("overall_accuracy", 0)
    insights.append(f"🎯 Overall System Accuracy: {overall_accuracy:.1%} ({metrics.get('problems_correct', 0)}/{metrics.get('successful', 0)} problems correct)")
    
    # Best performing LLM as Solver
    llm_win_rates = metrics.get("llm_win_rates", {})
    if llm_win_rates:
        best_solver = max(llm_win_rates.items(), key=lambda x: x[1])
        worst_solver = min(llm_win_rates.items(), key=lambda x: x[1])
        insights.append(f"🏆 Best Solver (Win Rate): {best_solver[0]} with {best_solver[1]:.1%} win rate ({metrics['llm_wins'].get(best_solver[0], 0)} wins)")
        insights.append(f"📉 Worst Solver (Win Rate): {worst_solver[0]} with {worst_solver[1]:.1%} win rate ({metrics['llm_wins'].get(worst_solver[0], 0)} wins)")
    
    # Best performing LLM by Accuracy
    llm_accuracy = metrics.get("llm_accuracy_rates", {})
    if llm_accuracy:
        best_accuracy = max(llm_accuracy.items(), key=lambda x: x[1])
        insights.append(f"🎯 Most Accurate Solver: {best_accuracy[0]} with {best_accuracy[1]:.1%} accuracy")
    
    # Best Judge
    llm_judge_accuracy = metrics.get("llm_judge_accuracy_rates", {})
    if llm_judge_accuracy:
        best_judge = max(llm_judge_accuracy.items(), key=lambda x: x[1])
        insights.append(f"⚖️ Best Judge (Accuracy): {best_judge[0]} with {best_judge[1]:.1%} accuracy in selecting correct solutions")
    
    # Most selected as Judge
    judge_selections = metrics.get("judge_selections", {})
    if judge_selections:
        most_selected = max(judge_selections.items(), key=lambda x: x[1])
        least_selected = min(judge_selections.items(), key=lambda x: x[1])
        insights.append(f"👑 Most Selected as Judge: {most_selected[0]} ({most_selected[1]} times, {most_selected[1]/metrics['successful']*100:.1f}% of problems)")
    
    # Participation statistics
    llm_participations = metrics.get("llm_solver_participations", {})
    if llm_participations:
        most_active = max(llm_participations.items(), key=lambda x: x[1])
        insights.append(f"💪 Most Active Solver: {most_active[0]} participated in {most_active[1]} problems")
    
    # Execution time insights
    if "avg_execution_time" in metrics:
        avg_time = metrics["avg_execution_time"]
        insights.append(f"⏱️ Average execution time: {avg_time:.2f} seconds per problem")
        
        if "max_execution_time" in metrics:
            max_time = metrics["max_execution_time"]
            if max_time > avg_time * 2:
                insights.append(f"⚠️ Some problems took significantly longer (max: {max_time:.2f}s vs avg: {avg_time:.2f}s)")
    
    # Problem type insights
    type_dist = metrics.get("type_distribution", {})
    if type_dist:
        most_common_type = max(type_dist.items(), key=lambda x: x[1])
        insights.append(f"📚 Most common problem type: {most_common_type[0]} ({most_common_type[1]} problems)")
    
    # Name uniqueness insights
    unique_rate = metrics.get("unique_name_rate", 1.0)
    if unique_rate < 0.9:
        duplicates = metrics.get("duplicate_names", {})
        insights.append(f"⚠️ Agent name uniqueness: {unique_rate:.1%} unique names. "
                       f"Found {len(duplicates)} duplicate names (disambiguation applied)")
    else:
        insights.append(f"✓ Agent names are highly unique: {unique_rate:.1%} unique names")
    
    # Error insights
    if metrics["failed"] > 0:
        error_types = metrics.get("error_types", {})
        if error_types:
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            insights.append(f"⚠️ Most common error type: {most_common_error[0]} ({most_common_error[1]} occurrences)")
    
    # Performance by type
    if "judge_by_type" in metrics:
        judge_by_type = metrics["judge_by_type"]
        insights.append(f"📈 Judge selection varies by problem type, showing adaptive behavior")
    
    return insights


def create_visualizations(metrics: Dict[str, Any], results: List[Dict[str, Any]], output_dir: Path):
    """
    Create comprehensive visualizations.
    
    Args:
        metrics: Calculated metrics
        results: All results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Success Rate Pie Chart
    fig, ax = plt.subplots(figsize=(8, 6))
    success_count = metrics["successful"]
    fail_count = metrics["failed"]
    ax.pie([success_count, fail_count], labels=["Success", "Failed"], autopct='%1.1f%%', 
           colors=['#2ecc71', '#e74c3c'], startangle=90)
    ax.set_title("Problem Execution Success Rate", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"success_rate_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Execution Time Distribution
    exec_times = [r["execution_time"] for r in results if r["success"]]
    if exec_times:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(exec_times, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(exec_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(exec_times):.2f}s')
        ax.axvline(np.median(exec_times), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(exec_times):.2f}s')
        ax.set_xlabel("Execution Time (seconds)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Problem Execution Times", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"execution_time_dist_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Judge Selection Bar Chart
    judge_selections = metrics.get("judge_selections", {})
    if judge_selections:
        fig, ax = plt.subplots(figsize=(10, 6))
        judges = list(judge_selections.keys())
        counts = list(judge_selections.values())
        bars = ax.bar(judges, counts, color=sns.color_palette("husl", len(judges)))
        ax.set_xlabel("Judge Agent ID", fontsize=12)
        ax.set_ylabel("Number of Selections", fontsize=12)
        ax.set_title("Judge Selection Frequency", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"judge_selection_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Problem Type Distribution
    type_dist = metrics.get("type_distribution", {})
    if type_dist:
        fig, ax = plt.subplots(figsize=(12, 6))
        types = list(type_dist.keys())
        counts = list(type_dist.values())
        bars = ax.bar(types, counts, color=sns.color_palette("Set2", len(types)))
        ax.set_xlabel("Problem Type", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Problem Type Distribution", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"problem_type_dist_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Judge Selection by Problem Type (Heatmap)
    judge_by_type = metrics.get("judge_by_type", {})
    if judge_by_type:
        # Create a matrix
        all_judges = set()
        for judges in judge_by_type.values():
            all_judges.update(judges.keys())
        all_judges = sorted(list(all_judges))
        
        matrix_data = []
        type_labels = []
        for prob_type, judges in judge_by_type.items():
            row = [judges.get(judge, 0) for judge in all_judges]
            matrix_data.append(row)
            type_labels.append(prob_type)
        
        if matrix_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(matrix_data, annot=True, fmt='d', cmap='YlOrRd', 
                       xticklabels=all_judges, yticklabels=type_labels,
                       cbar_kws={'label': 'Selection Count'})
            ax.set_xlabel("Judge Agent ID", fontsize=12)
            ax.set_ylabel("Problem Type", fontsize=12)
            ax.set_title("Judge Selection by Problem Type", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f"judge_by_type_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. Execution Time by Problem Type
    if results:
        df_viz = pd.DataFrame([
            {
                "type": r["type"],
                "execution_time": r["execution_time"],
                "success": r["success"]
            }
            for r in results
        ])
        
        if not df_viz.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            successful_df = df_viz[df_viz["success"]]
            if not successful_df.empty:
                sns.boxplot(data=successful_df, x="type", y="execution_time", ax=ax)
                ax.set_xlabel("Problem Type", fontsize=12)
                ax.set_ylabel("Execution Time (seconds)", fontsize=12)
                ax.set_title("Execution Time Distribution by Problem Type", fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_dir / f"exec_time_by_type_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    # 7. LLM Win Rates (as Solver)
    llm_win_rates = metrics.get("llm_win_rates", {})
    if llm_win_rates:
        fig, ax = plt.subplots(figsize=(10, 6))
        llms = list(llm_win_rates.keys())
        rates = [llm_win_rates[llm] * 100 for llm in llms]  # Convert to percentage
        bars = ax.bar(llms, rates, color=sns.color_palette("viridis", len(llms)))
        ax.set_xlabel("LLM Agent ID", fontsize=12)
        ax.set_ylabel("Win Rate (%)", fontsize=12)
        ax.set_title("LLM Win Rates as Solver", fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"llm_win_rates_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. LLM Accuracy Rates
    llm_accuracy = metrics.get("llm_accuracy_rates", {})
    if llm_accuracy:
        fig, ax = plt.subplots(figsize=(10, 6))
        llms = list(llm_accuracy.keys())
        accuracies = [llm_accuracy[llm] * 100 for llm in llms]
        bars = ax.bar(llms, accuracies, color=sns.color_palette("plasma", len(llms)))
        ax.set_xlabel("LLM Agent ID", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("LLM Accuracy Rates (Correct Solutions)", fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"llm_accuracy_rates_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 9. LLM Judge Accuracy
    llm_judge_accuracy = metrics.get("llm_judge_accuracy_rates", {})
    if llm_judge_accuracy:
        fig, ax = plt.subplots(figsize=(10, 6))
        llms = list(llm_judge_accuracy.keys())
        accuracies = [llm_judge_accuracy[llm] * 100 for llm in llms]
        bars = ax.bar(llms, accuracies, color=sns.color_palette("coolwarm", len(llms)))
        ax.set_xlabel("LLM Agent ID", fontsize=12)
        ax.set_ylabel("Judge Accuracy (%)", fontsize=12)
        ax.set_title("LLM Accuracy as Judge (Selecting Correct Solutions)", fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"llm_judge_accuracy_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 10. LLM Participation Comparison (Solver vs Judge)
    llm_solver_count = metrics.get("llm_as_solver_count", {})
    llm_judge_count = metrics.get("llm_as_judge_count", {})
    if llm_solver_count or llm_judge_count:
        all_llms = set(list(llm_solver_count.keys()) + list(llm_judge_count.keys()))
        if all_llms:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(all_llms))
            width = 0.35
            
            solver_counts = [llm_solver_count.get(llm, 0) for llm in all_llms]
            judge_counts = [llm_judge_count.get(llm, 0) for llm in all_llms]
            
            bars1 = ax.bar(x - width/2, solver_counts, width, label='As Solver', color='#3498db')
            bars2 = ax.bar(x + width/2, judge_counts, width, label='As Judge', color='#e74c3c')
            
            ax.set_xlabel("LLM Agent ID", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("LLM Participation: Solver vs Judge", fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(all_llms, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"llm_participation_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 11. Overall Accuracy Pie Chart
    problems_correct = metrics.get("problems_correct", 0)
    problems_incorrect = metrics.get("problems_incorrect", 0)
    if problems_correct + problems_incorrect > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie([problems_correct, problems_incorrect], 
               labels=["Correct", "Incorrect"], 
               autopct='%1.1f%%',
               colors=['#27ae60', '#c0392b'], 
               startangle=90)
        ax.set_title(f"Overall System Accuracy\n({problems_correct}/{problems_correct + problems_incorrect} problems)", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f"overall_accuracy_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 12. LLM Performance Comparison (Win Rate vs Accuracy)
    if llm_win_rates and llm_accuracy:
        common_llms = set(llm_win_rates.keys()) & set(llm_accuracy.keys())
        if common_llms:
            fig, ax = plt.subplots(figsize=(10, 6))
            llms = list(common_llms)
            win_rates = [llm_win_rates[llm] * 100 for llm in llms]
            accuracies = [llm_accuracy[llm] * 100 for llm in llms]
            
            x = np.arange(len(llms))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, win_rates, width, label='Win Rate (%)', color='#9b59b6', alpha=0.8)
            bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy (%)', color='#f39c12', alpha=0.8)
            
            ax.set_xlabel("LLM Agent ID", fontsize=12)
            ax.set_ylabel("Percentage (%)", fontsize=12)
            ax.set_title("LLM Performance: Win Rate vs Accuracy", fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(llms, rotation=45, ha='right')
            ax.set_ylim([0, 100])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"llm_performance_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    num_plots = 6 + sum([
        bool(llm_win_rates),
        bool(llm_accuracy),
        bool(llm_judge_accuracy),
        bool(llm_solver_count or llm_judge_count),
        bool(problems_correct + problems_incorrect > 0),
        bool(llm_win_rates and llm_accuracy)
    ])
    
    print(f"Generated {num_plots} visualization files in {output_dir}")


def save_insights(insights: List[str], metrics: Dict[str, Any], output_dir: Path):
    """
    Save insights to a text file.
    
    Args:
        insights: List of insight strings
        metrics: Calculated metrics
        output_dir: Directory to save insights
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    insights_path = output_dir / f"insights_{timestamp}.txt"
    
    with open(insights_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXECUTION INSIGHTS AND METRICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("KEY INSIGHTS:\n")
        f.write("-"*80 + "\n")
        for insight in insights:
            f.write(f"{insight}\n")
        
        f.write("\n\nDETAILED METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Problems: {metrics['total_problems']}\n")
        f.write(f"Successful: {metrics['successful']}\n")
        f.write(f"Failed: {metrics['failed']}\n")
        f.write(f"Success Rate: {metrics['success_rate']:.2%}\n\n")
        
        if "avg_execution_time" in metrics:
            f.write("Execution Time Statistics:\n")
            f.write(f"  Average: {metrics['avg_execution_time']:.2f}s\n")
            f.write(f"  Median: {metrics['median_execution_time']:.2f}s\n")
            f.write(f"  Min: {metrics['min_execution_time']:.2f}s\n")
            f.write(f"  Max: {metrics['max_execution_time']:.2f}s\n")
            f.write(f"  Std Dev: {metrics['std_execution_time']:.2f}s\n\n")
        
        f.write("Judge Selection Counts:\n")
        for judge, count in metrics.get("judge_selections", {}).items():
            f.write(f"  {judge}: {count}\n")
        
        f.write("\nProblem Type Distribution:\n")
        for prob_type, count in metrics.get("type_distribution", {}).items():
            f.write(f"  {prob_type}: {count}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("LLM PERFORMANCE METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Overall System Accuracy:\n")
        f.write(f"  Correct: {metrics.get('problems_correct', 0)}\n")
        f.write(f"  Incorrect: {metrics.get('problems_incorrect', 0)}\n")
        f.write(f"  Accuracy Rate: {metrics.get('overall_accuracy', 0):.2%}\n\n")
        
        f.write("LLM Win Rates (as Solver):\n")
        for llm_id, rate in sorted(metrics.get("llm_win_rates", {}).items(), key=lambda x: x[1], reverse=True):
            wins = metrics.get("llm_wins", {}).get(llm_id, 0)
            participations = metrics.get("llm_solver_participations", {}).get(llm_id, 0)
            f.write(f"  {llm_id}: {rate:.2%} ({wins}/{participations} wins)\n")
        
        f.write("\nLLM Accuracy Rates (Correct Solutions):\n")
        for llm_id, acc in sorted(metrics.get("llm_accuracy_rates", {}).items(), key=lambda x: x[1], reverse=True):
            stats = metrics.get("llm_accuracy", {}).get(llm_id, {"correct": 0, "total": 0})
            f.write(f"  {llm_id}: {acc:.2%} ({stats.get('correct', 0)}/{stats.get('total', 0)} correct)\n")
        
        f.write("\nLLM Judge Accuracy Rates:\n")
        for llm_id, acc in sorted(metrics.get("llm_judge_accuracy_rates", {}).items(), key=lambda x: x[1], reverse=True):
            stats = metrics.get("llm_judge_accuracy", {}).get(llm_id, {"correct": 0, "total": 0})
            f.write(f"  {llm_id}: {acc:.2%} ({stats.get('correct', 0)}/{stats.get('total', 0)} correct)\n")
        
        f.write("\nLLM Participation Counts:\n")
        f.write("  As Solver:\n")
        for llm_id, count in sorted(metrics.get("llm_as_solver_count", {}).items(), key=lambda x: x[1], reverse=True):
            f.write(f"    {llm_id}: {count}\n")
        f.write("  As Judge:\n")
        for llm_id, count in sorted(metrics.get("llm_as_judge_count", {}).items(), key=lambda x: x[1], reverse=True):
            f.write(f"    {llm_id}: {count}\n")


async def main():
    """
    Main async execution function.
    """
    print("\n" + "="*80)
    print("ASYNC PROBLEM RUNNER WITH METRICS AND VISUALIZATION")
    print("="*80 + "\n")
    
    # Configuration
    max_concurrent = 20  # Adjust based on API rate limits
    output_dir = Path("results")
    
    # Run all problems
    start_time = time.time()
    results = await run_all_problems_async(max_concurrent=max_concurrent)
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Completed {len(results)} problems in {total_time:.2f} seconds")
    print(f"{'='*80}\n")
    
    # Save results
    print("Saving results...")
    df_summary, json_path, pickle_path = save_results(results, output_dir)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    # Generate insights
    print("Generating insights...")
    insights = generate_insights(metrics, results)
    
    # Print insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    for insight in insights:
        print(f"  {insight}")
    print("="*80 + "\n")
    
    # Save insights
    save_insights(insights, metrics, output_dir)
    
    # Generate visualizations
    print("Generating visualizations...")
    try:
        create_visualizations(metrics, results, output_dir)
        print("✓ Visualizations generated successfully")
    except Exception as e:
        print(f"⚠ Warning: Visualization generation failed: {e}")
        print("  Results are still saved in JSON/pickle format")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Full results: {json_path}")
    print(f"  - Summary DataFrame: {pickle_path}")
    print(f"  - Insights: {output_dir / 'insights_*.txt'}")
    print(f"  - Visualizations: {output_dir / '*.png'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
