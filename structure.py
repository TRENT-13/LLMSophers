from utils.LLM_call import _load_llm_OPEN, _load_llm_deepseek
from functools import lru_cache
import io
import json
import os
import re
import sys
import operator
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Annotated


"""
Model of the project is the following: wer are going to use Langgraph for the project, 
first node would be about the judge choosing based on the question
second node would about LLM's solving questions
third node would critics, other LLM's, except the judge are going to evaluate the answer
fourth node would be the about the critics reading and refinment of the answer if critics were right
fifth and final node would be about the judge giving the final verdict based on the critics and the refined answer
"""

import pandas as pd
#at first create a dataset
@lru_cache(maxsize=1)
def dataset_creation():
    data = [
        # -------------------- MATH --------------------
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "given a system with two coupled masses connected by a spring, where gravity is neglected. The masses are suspended by massless, inextensible strings from a fixed horizontal beam. The system has two degrees of freedom. Find the normal modes and frequencies of small oscillations around the equilibrium position."
                "System Description:"
                "Two masses (both labeled 'm') suspended from a horizontal fixed beam"
                "Left mass: suspended by string at angle θ₁ from vertical"
                "Right mass: suspended by string at angle θ₂ from vertical"
                "The two masses are connected by a spring (shown as a coiled line between them)"
                "Length of each string is 'l' "
                "Task: Use Lagrange's formulation to derive the equations of motion for small oscillations around equilibrium, and find the normal modes and natural frequencies."
            ),
            "answer": "θ₁(t) = (θ₀ + ψ₀)/2 · cos ωt + (θ₀ - ψ₀)/2 · cos√(ω² + 2K)t"
        },
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Solve the system of first-order differential equations using Laplace transform:"
                "System of equations:"
                "dy₁/dt = -y₁ + 3y₂ + 4t"
                "dy₂/dt = 3y₁ - y₂ + cos(2t)"
                "Initial conditions:"
                "y₁(0) = 1, y₂(0) = 2"
                "Note: Write the Laplace transforms of the given functions in a table."
            ),
            "answer": "x₁(t) = [1 - 1]e⁻⁴ᵗ; x₂(t) = [1 1]e²ᵗ;"
        },
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Analyze a second-order linear differential system under a periodic external force."
            ),
            "answer": "General solution consists of homogeneous solution plus a particular solution determined by the forcing function."
        },
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Solve the initial value problem for the differential equation:"
                "Differential Equation:"
                "y'' + 9y = sin 3t - sin 3(t - 2π) u(t - 2π)"
                "Initial Conditions:"
                "y(0) = 0"
                "y'(0) = 0"

                "Where:"
                "u(t - 2π) is the unit step function (Heaviside function) shifted by 2π"
            ),
            "answer": "y(t) = (1/18)(sin(3t) - 3tcos(3t)) - (1/18)(sin(3(t - 2π)) - 3(t - 2π)cos(3(t - 2π)))u(t - 2π)"
        },
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Using Fourier series expansion of the function f, which satisfies periodic conditions with period T=8:"
                "Equation:"
                "3. y'' + y' + 4y = f(t),"
                "where:"
                "ft={1, -6≤x<-2 3, -2≤x≤2"
                "Charted/piecewise function"
            ),
            "answer": "a_n = (1/4)[2∫₂⁴ cos cos(nπt/4)dt + 2∫₀² 3 cos cos(nπt/4)dt]a_n = (2sin(πn) + 4sin(nπ/2))/(nπ)"
        },

        # -------------------- PHYSICS --------------------
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A capacitor is constructed from two square metallic plates separated by distance d. "
                "Charges Q and 2Q are placed on the plates and the power supply is removed. "
                "A dielectric with constant k is inserted a distance x into the capacitor.\n"
                "(a) Find the equivalent capacitance.\n"
                "(b) Find the stored energy.\n"
                "(c) Find the force on the dielectric.\n"
                "(d) Calculate the force for x = a/2, a = 5.00 cm, d = 2.00 mm, k = 4.50, "
                "and V = 2.00 × 10³ V."
            ),
            "answer": "The system is treated as two capacitors in parallel. (Numerical force value not explicitly stated in document.)"
        },
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "Circuit Description:"
                "- Network of capacitors between points a and b"
                "- NOT a simple series or parallel combination"
                "Topology (paths from a to b):"
                "1. Top path: 4.00 μF capacitor (direct connection a to b)"
                "2. Middle-left path: 2.00 μF capacitor (a to intermediate node)"
                "3. Middle-right path: 8.00 μF capacitor (intermediate node to b)"
                "4. Bottom path: 4.00 μF capacitor (a to lower intermediate node) "
                " connected to 2.00 μF capacitor (lower intermediate node to b)"
                "Visual structure:"
                "- Forms a bridge/mesh network"
                "- Multiple intermediate nodes create parallel pathways"
                "- Requires potential difference analysis between nodes"
            ),
            "answer": "3.00 mF"
        },
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A quantum simple harmonic oscillator consists of an electron bound by a restoring force "
                "with proportionality constant k = 8.99 N/m. "
                "What is the longest wavelength of light that can excite the oscillator?"
            ),
            "answer": "600 nm"
        },
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A particle of mass m is located in a two-dimensional potential well with absolutely impenetrable walls. Find:"
                "a) The smallest values of the particle's energy, if the sides of the well are l₁ and l₂."
                "b) The energy values at the first four levels, if the well is a square with side l."
            ),
            "answer": "1st level: n₁ = n₂ = 1 → π² = 9.87"
                "2nd level:"
                "n₁ = 1, n₂ = 2 } → 5/2 π² = 24.7"
                "n₁ = 2, n₂ = 1"
                "3rd level: n₁ = n₂ = 2 → 4π² = 39.5"
                "4th level:"
                "n₁ = 1, n₂ = 3 } → 5π² = 49.3"
                "n₁ = 3, n₂ = 1"
        },
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A person stands at the edge of a pond and observes a stone on the bottom. The depth of the pond is h. How far from the water's surface is the virtual image of the stone if the ray of vision makes an angle theta with the normal to the water surface?"
            ),
            "answer": "\frac{\hbar n^3 \cos^3 \theta}{\left(n^3 - \sin^3 \theta\right)^{3/2}}"
        }
    ]

    df = pd.DataFrame(data)
    df.to_pickle("data")
    return df

@lru_cache(maxsize=1)
def load_dataset():
    import pandas as pd

    try:
        df = pd.read_pickle("data")
    except FileNotFoundError:
        df = dataset_creation()
    return df


class AgentState(TypedDict):
    """State that flows through the graph"""
    question: str
    question_type: str  # math, physics, etc.
    field: str  # STEM, etc.
    
    # Agent identities
    agent_names: dict  # Maps agent_id (GPT-1, etc.) to chosen name (e.g., "Augustus")
    
    # Judge election
    election_responses: list[dict]  # Each agent's initial response/argument
    judge_votes: dict  # Vote tallies by chosen name
    elected_judge_name: str  # The chosen name of elected judge
    elected_judge_id: str  # The ID (GPT-1, etc.) of elected judge
    solver_ids: list[str]  # List of non-judge agent IDs
    solver_names: list[str]  # List of non-judge agent names
    
    # Solving phase
    solver_answers: list[dict]  # Multiple solvers provide answers
    best_answer: str  # Best answer selected from solvers
    
    # Critique phase
    critic_feedback: list[str]  # Multiple critics provide feedback
    
    # Final phase
    refined_answer: str
    final_verdict: str
    confidence_score: float
    
    messages: Annotated[list, operator.add]  # Accumulate messages



gpt1 = _load_llm_OPEN(model_name="gpt-4o-mini", temperature=0.0)
gpt2 = _load_llm_OPEN(model_name="gpt-5-mini", temperature=0.0)
deepseek1 = _load_llm_deepseek(model_name="DeepSeek-V3.2", temperature=0)
deepseek2 = _load_llm_deepseek(model_name="deepseek-reasoner", temperature=0)

LLM_AGENTS = {
    "GPT-1": llm_gpt1,
    "GPT-2": llm_gpt2,
    "DeepSeek-1": llm_deepseek1,
    "DeepSeek-2": llm_deepseek2
}



# lets name the agents
def agent_naming() -> dict:
    """
    Phase 1: Each agent chooses a distinguished name for themselves
    
    Args:
        question: The question that will be solved
    
    Returns:
        dict: Maps agent_id to chosen name
    """
    naming_prompt = f"""you are the  agent, name yourself whatever name you like, like George, Napoleon,Augustus...
    Choose a distinguished name for yourself - something memorable
    RESPOND WITH ONLY 1 WORD: [Your chosen name]
    """
    agent_names = {}

    for agent_id, llm in LLM_AGENTS.items():
        prompt = naming_prompt.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)]) 
        name = response.content
        agent_names[agent_id] = name
        print(f"  {agent_id:12} → {chosen_name}")

    return agent_names


def run_agent_deliberation(question: str, agent_names: dict) -> list:
    """
    Phase 2: Each agent analyzes the question and argues who should be judge
    
    Args:
        question: The question to be solved
        agent_names: Maps agent_id to chosen name
    
    Returns:
        list: Each agent's deliberation response
    """
    agents_list = "\n".join([f"  • {agent_names[aid]} (Agent {aid})" for aid in LLM_AGENTS.keys()])
    deliberation_prompt = """You are {my_name} (Agent {my_id}), participating in a democratic process to elect a judge.

    CONTEXT: You and three other AI agents will collaboratively solve this problem. First, you must elect one agent to serve as the judge who will make the final verdict. The other three agents will work as solvers.

    QUESTION TO SOLVE:
    {question}

    PARTICIPATING AGENTS:
    {agents_list}

    YOUR TASK:
    1. Analyze what specific qualities and expertise the judge needs for THIS question
    2. Recommend which agent should be the judge (you may recommend yourself or another)
    3. Provide detailed reasoning for your recommendation

    NOTE: You will vote AFTER seeing all arguments. For now, just make your case.

    RESPOND IN THIS FORMAT:
    Analysis: [What makes this question unique? What expertise does the judge need?]
    Recommendation: [Name of agent who should be judge]
    Reasoning: [Why this agent is the best choice for judging THIS specific question]"""
    deliberation_responses = []
    for agent_id, llm in LLM_AGENTS.items():
        prompt = deliberation_prompt.format(
            my_name=agent_names[agent_id],
            my_id=agent_id,
            question=question,
            agents_list=agents_list
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        deliberation_responses.append({
            "agent_id": agent_id,
            "agent_name": agent_names[agent_id],
            "response": response.content
        })
        print(f"  ✓ {agent_names[agent_id]} completed deliberation")
    return deliberation_responses



def run_agent_voting(question: str, agent_names: dict, deliberation_responses: list) -> tuple:
    """
    Phase 3: Each agent sees all deliberations and casts their vote
    
    Args:
        question: The question to be solved
        agent_names: Maps agent_id to chosen name
        deliberation_responses: All agents' deliberation arguments
    
    Returns:
        tuple: (votes dict, vote_details list)
    """
    # Format all deliberations for sharing
    all_arguments = "\n\n".join([
        f"--- {resp['agent_name']}'s Argument ---\n{resp['response']}"
        for resp in deliberation_responses
    ])
    
    voting_prompt = """You are {my_name} (Agent {my_id}). You've just heard all arguments about who should be the judge.

    CONTEXT: You are electing one agent to serve as judge for this problem. The judge will make the final verdict after three solvers work on the solution. This is a critical decision.

    QUESTION TO SOLVE:
    {question}

    ALL ARGUMENTS:
    {all_arguments}

    YOUR TASK:
    Now that you've heard everyone's arguments, cast your vote for who should be the judge. Consider:
    - Who made the most compelling argument?
    - Which agent has the right qualities for THIS specific question?
    - Who would be the fairest and most capable judge?

    RESPOND WITH ONLY:
    Vote: [Name of agent you're voting for]
    Justification: [One sentence explaining why you chose this agent]"""
        
    votes = {agent_names[aid]: 0 for aid in LLM_AGENTS.keys()}
    vote_details = []
    
    print("="*80)
    print("PHASE 3: VOTING")
    print("="*80)
    
    for agent_id, llm in LLM_AGENTS.items():
        prompt = voting_prompt.format(
            my_name=agent_names[agent_id],
            my_id=agent_id,
            question=question,
            all_arguments=all_arguments
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse vote - match any of the agent names
        vote_cast = None
        response_lower = response.content.lower()
        for aid, name in agent_names.items():
            if name.lower() in response_lower:
                vote_cast = name
                votes[name] += 1
                break
        
        # If no match, try to extract from "Vote:" line
        if not vote_cast:
            vote_match = re.search(r'Vote:\s*([A-Za-z]+)', response.content)
            if vote_match:
                potential_name = vote_match.group(1)
                for aid, name in agent_names.items():
                    if name.lower() == potential_name.lower():
                        vote_cast = name
                        votes[name] += 1
                        break
        
        vote_details.append({
            "voter": agent_names[agent_id],
            "voted_for": vote_cast or "Unknown",
            "justification": response.content
        })
        print(f"  🗳️  {agent_names[agent_id]:12} voted for: {vote_cast or 'Unknown'}")
    
    print("="*80 + "\n")
    return votes, vote_details


def judge_election_node(state: WorkflowState) -> WorkflowState:
    """
    Democratic judge election with three phases:
    1. Agent naming
    2. Agent deliberation
    3. Agent voting
    """
    question = state["question"]
    
    # Phase 1: Agents choose their names
    agent_names = run_agent_naming(question)
    
    # Phase 2: Agents make arguments for who should be judge
    deliberation_responses = run_agent_deliberation(question, agent_names)
    
    # Phase 3: Agents see all arguments and vote
    votes, vote_details = run_agent_voting(question, agent_names, deliberation_responses)
    
    # Determine winner
    elected_judge_name = max(votes, key=votes.get)
    elected_judge_id = [aid for aid, name in agent_names.items() if name == elected_judge_name][0]
    
    # Identify solvers (non-judge agents)
    solver_ids = [aid for aid in LLM_AGENTS.keys() if aid != elected_judge_id]
    solver_names = [agent_names[aid] for aid in solver_ids]
    
    # Parse question type from first deliberation
    field = "STEM"
    question_type = "general"
    first_analysis = deliberation_responses[0]["response"].lower()
    if "math" in first_analysis:
        question_type = "math"
    elif "physic" in first_analysis:
        question_type = "physics"
    elif "chemistry" in first_analysis or "chemical" in first_analysis:
        question_type = "chemistry"
    
    print("="*80)
    print(f" ELECTED JUDGE: {elected_judge_name} ({elected_judge_id})")
    print(f"Vote Results: {votes}")
    print(f"Solvers: {', '.join(solver_names)}")
    print("="*80 + "\n")
    
    return {
        **state,
        "agent_names": agent_names,
        "election_responses": deliberation_responses,
        "judge_votes": votes,
        "elected_judge_name": elected_judge_name,
        "elected_judge_id": elected_judge_id,
        "solver_ids": solver_ids,
        "solver_names": solver_names,
        "field": field,
        "question_type": question_type,
        "messages": [
            AIMessage(
                content=f"Judge Election Complete:\n"
                        f"Elected: {elected_judge_name} ({elected_judge_id})\n"
                        f"Votes: {votes}\n"
                        f"Solvers: {', '.join(solver_names)}",
                name="election_system"
            )
        ]
    }