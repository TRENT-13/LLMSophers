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
    
    # Judge election
    judge_arguments: list[dict]  # Each LLM's argument for who should be judge
    elected_judge: str  # Which LLM was elected as judge
    judge_votes: dict  # Vote tallies
    
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



