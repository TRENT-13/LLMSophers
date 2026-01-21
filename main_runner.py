"""
Main execution script for the LLM Collaborative Problem Solver
"""
from structure import (
    load_dataset, 
    judge_election_node, 
    solver_node,
    AgentState
)
import random


def main():
    """
    Main execution function - demonstrates the workflow
    """
    print("\n" + "="*80)
    print("=" + " "*78 + "=")
    print("=" + " "*20 + "LLM COLLABORATIVE PROBLEM SOLVER" + " "*26 + "=")
    print("=" + " "*78 + "=")
    print("="*80 + "\n")
    
    # Load dataset and pick a random question
    df = load_dataset()
    
    print("Dataset loaded with {} questions".format(len(df)))
    print("\nAvailable question types:")
    for idx, row in df.iterrows():
        print(f"  [{idx}] {row['type']:10} - {row['question'][:80]}...")
    
    # Pick a random question
    selected_idx = random.randint(0, len(df) - 1)
    selected_question = df.iloc[selected_idx]
    
    print("\n" + "="*80)
    print(f"RANDOMLY SELECTED QUESTION #{selected_idx}")
    print("="*80)
    print(f"Field: {selected_question['field']}")
    print(f"Type: {selected_question['type']}")
    print(f"\nQuestion:\n{selected_question['question']}")
    print(f"\nExpected Answer:\n{selected_question['answer']}")
    print("="*80 + "\n")
    
    # Initialize state
    initial_state: AgentState = {
        "question": selected_question['question'],
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
    
    # Run Node 1: Judge Election
    try:
        print("Starting Node 1: Judge Election...")
        state_after_election = judge_election_node(initial_state)
        
        print("\n" + "="*80)
        print("NODE 1 COMPLETE: Judge Election")
        print("="*80)
        print(f"Elected Judge: {state_after_election['elected_judge_name']} ({state_after_election['elected_judge_id']})")
        print(f"Solvers: {', '.join(state_after_election['solver_names'])}")
        print("="*80 + "\n")
        
        # Run Node 2: Solving Phase
        print("Starting Node 2: Solving Phase...")
        state_after_solving = solver_node(state_after_election)
        
        print("\n" + "="*80)
        print("NODE 2 COMPLETE: Solving Phase")
        print("="*80)
        print(f"Number of solutions: {len(state_after_solving['solver_answers'])}")
        print("\n--- SOLUTIONS SUMMARY ---")
        for sol in state_after_solving['solver_answers']:
            print(f"\n{sol['agent_name']}'s Solution:")
            print("-" * 40)
            # Print first 300 chars of solution
            solution_preview = sol['answer'][:300] + "..." if len(sol['answer']) > 300 else sol['answer']
            print(solution_preview)
        print("\n" + "="*80 + "\n")
        
        print("\n" + "="*80)
        print("WORKFLOW DEMONSTRATION COMPLETE")
        print("="*80)
        
        return state_after_solving
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()