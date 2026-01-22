"""
Main execution script for the LLM Collaborative Problem Solver
"""
from structure import (
    load_dataset, 
    judge_election_node, 
    solver_node,
    critic_node,
    refinement_node,
    final_verdict_node,
    AgentState
)
import random


def main():
    """
    Main execution function - demonstrates the full collaborative workflow
    """
    print("\n" + "="*80)
    print("=" + " "*78 + "=")
    print("=" + " "*20 + "LLM COLLABORATIVE PROBLEM SOLVER" + " "*26 + "=")
    print("=" + " "*78 + "=")
    print("="*80 + "\n")
    
    # Load dataset and pick a random question
    df = load_dataset()
    
    print("Dataset loaded with {} questions".format(len(df)))
    
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
    state: AgentState = {
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
    
    try:
        # --- Node 1: Judge Election ---
        print(">>> Starting Stage 1: Judge Election...")
        state = judge_election_node(state)
        
        print(f"\n[Election Result] Judge: {state['elected_judge_name']} | Solvers: {', '.join(state['solver_names'])}")
        
        # --- Node 2: Solving Phase ---
        print("\n>>> Starting Stage 2: Solving Phase...")
        state = solver_node(state)
        
        print("\n[Solving Complete] Initial solutions generated.")
        
        # --- Node 3: Peer Review ---
        print("\n>>> Starting Stage 3: Peer Review Phase...")
        state = critic_node(state)
        
        print(f"\n[Review Complete] Generated {len(state['critic_feedback'])} critiques.")
        
        # --- Node 4: Refinement ---
        print("\n>>> Starting Stage 4: Refinement Phase...")
        state = refinement_node(state)
        
        print("\n[Refinement Complete] Solvers have updated their answers.")
        
        # --- Node 5: Final Verdict ---
        print("\n>>> Starting Stage 5: Final Judgment...")
        state = final_verdict_node(state)
        
        print("\n" + "="*80)
        print("FINAL RESULT")
        print("="*80)
        print(state['final_verdict'])
        print("="*80 + "\n")
        
        return state
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
