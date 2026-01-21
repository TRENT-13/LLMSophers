from utils.LLM_call import _load_llm_OPEN, _load_llm_deepseek, _load_llm_OPEN5
from functools import lru_cache
import io
import json
import os
import re
import sys
import operator
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Annotated
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

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
            "answer": r"\frac{\hbar n^3 \cos^3 \theta}{\left(n^3 - \sin^3 \theta\right)^{3/2}}"
        },
        #   -------------------- LOGIC  --------------------   
        {
        "field": "STEM",
        "type": "logic",
        "question": "On an island of Knights (always tell truth) and Knaves (always lie), you meet three inhabitants: A, B, and C. A says 'B is a knave'. B says 'A and C are of the same type'. C says 'I have the same type as B'. Determine the type of each inhabitant.",
        "answer": "A is a Knight, B is a Knave, C is a Knave."
        },
        {
            "field": "STEM",
            "type": "logic",
            "question": "Five friends (Alice, Bob, Charlie, David, Eve) represent 5 different colors (Red, Blue, Green, Yellow, White) and own 5 different pets (Dog, Cat, Fish, Bird, Snake). 1. The Green owner has a Snake. 2. Alice does not own the Red color. 3. Bob owns the Dog. 4. The White color owner is immediately to the right of the Green owner. 5. David owns the Cat and is next to the Blue owner. 6. Eve is on the far left. 7. The Bird owner is in the middle spot. 8. The Red owner is next to the Dog owner. Who owns the Fish?",
            "answer": "Eve owns the Fish."
        },
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Two distinct integers are chosen from the set {2, 3, ..., 99}. One integer is given to Alice (product P) and the other to Bob (sum S). Alice says 'I don't know the numbers'. Bob says 'I knew you didn't know'. Alice says 'Now I know the numbers'. Bob says 'Now I know the numbers too'. What are the two numbers? (This is a classic incomplete information game known as the Sum and Product Puzzle).",
            "answer": "4 and 13"
        },
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Consider a Second-Price Sealed-Bid Auction (Vickrey Auction) for an antique vase. You value the vase at $500. Your opponent's valuation is unknown to you but is drawn from a uniform distribution between $0 and $1000. What is your optimal bidding strategy (b) to maximize expected utility?",
            "answer": "Bid exactly your valuation: b = $500."
        },
        {
            "field": "STEM",
            "type": "logic",
            "question": "Three logic professors (A, B, C) are shown 5 stamps: 2 red and 3 green. They are blindfolded, and one stamp is pasted on each of their foreheads. The remaining 2 are hidden. When blindfolds are removed, A is asked if she knows her color. She says 'No'. B is asked; he says 'No'. C is asked; she says 'Yes'. What color is C's stamp and why?",
            "answer": "Green. C deduces this because if C were Red, B would have seen a Red on C. If A also had Red, B would have known B was Green immediately (since there are only 2 Reds). Even if A had Green, B would have realized that if C was Red, B must be Green to prevent A from knowing. Since B didn't know, C cannot be Red."
        },

        #  -------------------- GAME THEORY  --------------------
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Five rational pirates (A, B, C, D, E) find 100 gold coins. They must propose a distribution plan. The strict order of seniority is A > B > C > D > E. The most senior pirate proposes a split. All vote (including the proposer). If 50% or more vote 'yes', the plan passes. Otherwise, the proposer is thrown overboard and the next senior proposes. Pirates maximize their gold first, and prefer survival second. What is the optimal proposal for Pirate A?",
            "answer": "A: 98, B: 0, C: 1, D: 0, E: 1"
        },
        {
            "field": "STEM",
            "type": "logic",
            "question": "A census taker approaches a house and asks about the ages of the three children inside. The woman says, 'The product of their ages is 36. The sum of their ages is the house number next door.' The census taker looks at the house number but says, 'I still need more information.' The woman replies, 'The oldest is sleeping upstairs.' What are the ages of the children?",
            "answer": "9, 2, and 2."
        },
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "In a game of Nim, there are three heaps of coins with sizes 3, 4, and 5. Two players take turns removing any number of coins from a single heap. The player to take the last coin wins. Is the current position a winning or losing position for the first player, and what is the 'Nim-sum' of this configuration?",
            "answer": "Winning position. Nim-sum = 3 XOR 4 XOR 5 = 011 ^ 100 ^ 101 = 010 (binary) = 2. Winning move is to reduce a heap to make Nim-sum 0 (e.g., change heap of 3 to 1)."
        },
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Consider a Cournot Duopoly where two firms produce identical goods. The inverse demand function is P = 120 - Q, where Q = q1 + q2. The marginal cost for both firms is constant at 0. Find the Nash Equilibrium quantities (q1, q2) for the two firms.",
            "answer": "q1 = 40, q2 = 40"
        },
        {
            "field": "STEM",
            "type": "logic",
            "question": "You have 12 coins that look identical. One is counterfeit and weighs slightly different (heavier or lighter, you don't know) than the others. You have a balance scale and can use it exactly 3 times. Construct a strategy to isolate the fake coin and determine if it is heavier or lighter.",
            "answer": "Weigh 4 vs 4. Case 1 (Equal): Fake is in the remaining 4. Weigh 3 normal vs 3 remaining. If equal, last one is fake (weigh against normal to find bias). If unequal, you know bias, weigh 1 vs 1. Case 2 (Unequal): We now know the group containing the fake and potential biases. Perform a mixed swap (e.g., 3 from heavy side, 3 from light side, etc.) to isolate."
        },
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Two criminals are arrested. If both remain silent, they get 1 year each. If one betrays (confesses) and the other is silent, the betrayer goes free and the silent one gets 3 years. If both betray, they get 2 years each. This is a one-shot game. Identify the Nash Equilibrium strategy profile.",
            "answer": "{Betray, Betray}"
        },
        {
            "field": "STEM",
            "type": "logic",
            "question": "Four people need to cross a rickety bridge at night. They have one flashlight. The bridge holds at most two people. Any party crossing must carry the flashlight. The people walk at different speeds: 1 min, 2 mins, 5 mins, and 10 mins. When two people walk together, they walk at the slower person's speed. What is the minimum time required for all four to cross?",
            "answer": "17 minutes. (Order: 1&2 cross (2), 1 returns (1), 5&10 cross (10), 2 returns (2), 1&2 cross (2). Total: 2+1+10+2+2 = 17)."
        },
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Consider the 'Battle of the Sexes' game. Husband prefers Opera (Payoff: H=3, W=1), Wife prefers Football (Payoff: H=1, W=3). If they go to different places, both get 0. Find the Mixed Strategy Nash Equilibrium probability (p) that the Husband goes to the Opera.",
            "answer": "p = 3/4 (Husband chooses Opera with probability 3/4, Wife chooses Football with probability 3/4)."
        },
        {
            "field": "STEM",
            "type": "logic",
            "question": "Three gods A, B, and C are called True, False, and Random. True always speaks truth, False always lies, but Random answers randomly. You must determine who is who by asking 3 yes/no questions. Each question is directed at only one god. The gods understand English but answer in their own language: 'da' and 'ja', but you don't know which means yes and which means no. What is the first question you should ask to eliminate 'Random' as a possibility for one specific candidate?",
            "answer": "Ask God B: 'If I asked you 'Is God A Random?', would you say 'ja'?' (If B answers 'ja', then C is not Random. If B answers 'da', then A is not Random)."
        },
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Two players play a game where they take turns placing a penny on a round table. The coins cannot overlap and must not hang off the edge. The last player to fit a coin on the table wins. The table is finite. What is the winning strategy for Player 1?",
            "answer": "Player 1 places the first coin exactly in the center of the table. For every subsequent move by Player 2 at position P, Player 1 places a coin at position -P (symmetrically opposite across the center)."
        },
    ]

    df = pd.DataFrame(data)
    df.to_pickle("dataset.pkl")
    return df

@lru_cache(maxsize=1)
def load_dataset():
    import pandas as pd

    try:
        df = pd.read_pickle("dataset.pkl")
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



gpt1 = _load_llm_OPEN5(model_name="gpt-5") # do not use gpt 5 nano
gpt2 = _load_llm_OPEN(model_name="gpt-4o-mini", temperature=0.0)
deepseek1 = _load_llm_deepseek(model_name="deepseek-chat", temperature=0)
deepseek2 = _load_llm_deepseek(model_name="deepseek-reasoner", temperature=0.3)

LLM_AGENTS = {
    "GPT-1": gpt1,
    "GPT-2": gpt2,
    "DeepSeek-1": deepseek1,
    "DeepSeek-2": deepseek2
}



# lets name the agents
def agent_naming() -> dict:
    """
    Phase 1: Each agent chooses a distinguished name for themselves
    
    Returns:
        dict: Maps agent_id to chosen name
    """
    naming_prompt = """You are an AI agent. Choose a distinguished name for yourself - something memorable like George, Napoleon, Augustus, etc.
    
    RESPOND WITH ONLY ONE WORD: [Your chosen name]
    """
    agent_names = {}

    print("="*80)
    print("PHASE 1: AGENT NAMING")
    print("="*80)
    
    for agent_id, llm in LLM_AGENTS.items():
        response = llm.invoke([HumanMessage(content=naming_prompt)]) 
        name = response.content.strip()
        agent_names[agent_id] = name
        print(f"  {agent_id:12} → {name}")
        print(f"     Response: {response.content}")
    
    print("="*80 + "\n")
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
        print(f"\n  ✓ {agent_names[agent_id]}'s Deliberation:")
        print(f"  {'-'*60}")
        print(f"  {response.content}")
        print(f"  {'-'*60}")
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
        print(f"\n  🗳️  {agent_names[agent_id]:12} voted for: {vote_cast or 'Unknown'}")
        print(f"     Response: {response.content[:200]}..." if len(response.content) > 200 else f"     Response: {response.content}")
    
    print("="*80 + "\n")
    return votes, vote_details


def judge_election_node(state: AgentState) -> AgentState:
    """
    Democratic judge election with three phases:
    1. Agent naming
    2. Agent deliberation
    3. Agent voting
    """
    question = state["question"]
    
    print("\n" + "#"*80)
    print("# NODE 1: JUDGE ELECTION")
    print("#"*80 + "\n")
    
    # Phase 1: Agents choose their names
    agent_names = agent_naming()
    
    # Phase 2: Agents make arguments for who should be judge
    print("="*80)
    print("PHASE 2: DELIBERATION")
    print("="*80)
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



def solver_node(state: AgentState) -> AgentState:
    """
    non-judge agents solve the questions
    """
    print("\n" + "#"*80)
    print("# NODE 2: SOLVING PHASE")
    print("#"*80 + "\n")
    
    question = state["question"]
    question_type = state["question_type"]
    field = state["field"]
    solver_ids = state["solver_ids"]
    agent_names = state["agent_names"]
    elected_judge_name = state["elected_judge_name"]

    solver_answers = []
    for agent_id in solver_ids:
        agent_name = agent_names[agent_id]
        solution = generate_solver_solution(
            agent_id, agent_name, question, 
            question_type, field, elected_judge_name
        )
        solver_answers.append({
            "agent_id": agent_id,
            "agent_name": agent_name,
            "answer": solution
        })
        print(f"  ✓ {agent_name} completed solution")
    
    combined_answers = "\n\n---\n\n".join([
        f"Solution from {sol['agent_name']}:\n{sol['answer']}" 
        for sol in solver_answers
    ])
    
    print("="*80 + "\n")
    
    return {
        **state,
        "solver_answers": solver_answers,
        "best_answer": combined_answers,
        "messages": [
            AIMessage(
                content=f"Solutions from {', '.join([s['agent_name'] for s in solver_answers])}",
                name="solver_team"
            )
        ]
    }


def generate_solver_solution(agent_id: str, agent_name: str, question: str, 
                            question_type: str, field: str, judge_name: str) -> str:
    """
    Generate a solution from a single solver agent
    
    Args:
        agent_id: The agent's ID
        agent_name: The agent's chosen name
        question: The problem to solve
        question_type: Type of question (math, physics, etc.)
        field: Field of study
        judge_name: Name of the elected judge
    
    Returns:
        str: The agent's solution
    """
    llm = LLM_AGENTS[agent_id]
    
    solver_prompt = """You are {agent_name}, one of three solvers working on this problem.

        CONTEXT: {judge_name} was elected as the judge. You and two other agents are the solvers. After all three of you provide solutions, critics will evaluate them, and {judge_name} will make the final verdict.

        PROBLEM TYPE: {question_type} in {field}

        QUESTION:
        {question}

        YOUR TASK:
        Provide a complete, detailed solution:
        1. Explain your approach and reasoning
        2. Show all steps and calculations
        3. Clearly mark your final answer

        Be thorough and rigorous in your work."""
    
    prompt = solver_prompt.format(
        agent_name=agent_name,
        judge_name=judge_name,
        question_type=question_type,
        field=field,
        question=question
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print(f"\n   {agent_name}'s Solution:")
    print(f"  {'-'*70}")
    # Show first 500 chars during execution
    preview = response.content[:500] + "..." if len(response.content) > 500 else response.content
    print(f"  {preview}")
    print(f"  {'-'*70}")
    
    return response.content