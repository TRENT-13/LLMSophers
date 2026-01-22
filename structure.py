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
# let's make the dataset
@lru_cache(maxsize=1)
def dataset_creation():
    data = [
        # ==================================================================================
        # MATHEMATICS & NUMBER THEORY
        # ==================================================================================
        
        # infinite power tower limits
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Solve for x in the infinite power tower: x^(x^(x^...)) = 2. "
                "Then, attempt to solve for x in x^(x^(x^...)) = 4. "
                "Explain why the second case has no real solution despite algebraic manipulation suggesting x = sqrt(2)."
            ),
            "answer": "Case 1: x = sqrt(2). Case 2: No solution. The function f(t) = x^t has a stable fixed point only if e^(-e) <= x <= e^(1/e). For equal to 4, the required x would be outside the convergence range."
        },
        # rump's royal pain - floating point nightmare
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Compute the value of the polynomial: "
                "P(a, b) = 333.75b^6 + a^2(11a^2b^2 - b^6 - 121b^4 - 2) + 5.5b^8 + a/(2b) "
                "for a = 77617 and b = 33096. "
                "Warning: Standard floating point arithmetic may yield a result around -1.18 x 10^21 or 1.17 x 10^37. "
                "Provide the correct value to at least 2 decimal places and explain why standard calculators fail."
            ),
            "answer": "-0.83 (approx). The terms cancel out massively, leaving a tiny residual that is lost in floating point precision error (catastrophic cancellation)."
        },
        # borwein integrals - pattern breaking
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Evaluate the integral: I = Integral from 0 to infinity of [ Product_{k=0 to 14} sinc(x/(2k+1)) ] dx. "
                "Is the value exactly pi/2? "
                "Note: sinc(t) = sin(t)/t. The pattern holds that the integral is pi/2 for k=0 up to k=13."
            ),
            "answer": "No. It is slightly less than pi/2. The pattern breaks at the 15th term (k=14) due to the sum of reciprocals exceeding 1 (Borwein Integrals)."
        },
        # frame-stewart algorithm
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Consider the Tower of Hanoi problem with n disks and k=4 pegs. "
                "Using the Frame-Stewart algorithm, determine the minimum number of moves required to move n=8 disks. "
                "Show the recurrence relation used."
            ),
            "answer": "Recurrence: T(n, k) = min { 2*T(r, k) + T(n-r, k-1) } for 1 <= r < n. For n=8, k=4, the optimal split is r=4 or 5? Minimal moves is 33."
        },
        # hilbert matrix condition number
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Calculate the condition number (infinity norm) of the 4x4 Hilbert Matrix H, where H_ij = 1/(i+j-1). "
                "Explain why inverting this matrix using standard floating point arithmetic is prone to large errors."
            ),
            "answer": "The Hilbert matrix is notoriously ill-conditioned. For n=4, cond(H) is approx 28,000+. Small perturbations in input (rounding errors) lead to massive errors in the solution."
        },
        # dirichlet function integration
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Let f(x) be the function: f(x) = 1 if x is rational, and f(x) = 0 if x is irrational. "
                "1. Evaluate the Riemann integral of f(x) over [0, 1]. "
                "2. Evaluate the Lebesgue integral of f(x) over [0, 1]. "
                "Explain precisely why the results differ (or don't exist) based on the definitions of these integrals."
            ),
            "answer": "Riemann integral is undefined (upper sum 1, lower sum 0). Lebesgue integral is 0 because the set of rationals has measure zero."
        },
        # banach-tarski paradox
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Explain how it is possible to decompose a solid unit ball into a finite number of disjoint sets and reassemble them into TWO solid unit balls. "
                "Why doesn't this apply to physical matter (e.g., a gold ball)?"
            ),
            "answer": "It relies on non-measurable sets constructed via the Axiom of Choice. Physical matter is composed of discrete atoms, which cannot be split infinitely like mathematical points, so the volume-doubling transformation is impossible in reality."
        },
        # quintic solvability
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Determine if the quintic equation x^5 - 4x + 2 = 0 is solvable by radicals. "
                "Use Eisenstein's criterion to check irreducibility, and describe the properties of its Galois group that determine solvability."
            ),
            "answer": "Irreducible by Eisenstein (p=2). It has 3 real roots and 2 complex. The Galois group is S5 (symmetric group), which is not a solvable group. Thus, not solvable by radicals."
        },
        # hydra game - ordinal arithmetic
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Consider a rooted tree (Hydra). at step n, the player cuts a head (leaf node). "
                "The Hydra grows N copies of the subtree attached to the parent of the cut node. "
                "Prove using Ordinal Arithmetic (epsilon-zero) that the Hydra eventually dies for ANY strategy. "
                "Why can't this be proven in Peano Arithmetic?"
            ),
            "answer": "Every state of the Hydra corresponds to an ordinal number < epsilon_0. Each move strictly decreases this ordinal. Since the ordinals are well-ordered, there is no infinite descending sequence (termination is guaranteed). Peano Arithmetic cannot prove this because the principle of induction up to epsilon_0 is not expressible/provable within PA (Godel/Gentzen)."
        },
        # devil's integral
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Evaluate the integral from -infinity to infinity of exp(i * x^2) dx. "
                "Now, evaluate the integral from 0 to infinity of x^(-1/2) * exp(-x) * cos(x^2) dx. "
                "Identify the correct contour and branch cuts required."
            ),
            "answer": "First part is Gaussian/Fresnel integral: sqrt(i * pi). Second part requires a 'keyhole' contour or specific substitution. The 'Devil' detail is handling the convergence at 0 and infinity simultaneously with the oscillation."
        },
        # moving sofa problem
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "What is the exact value of the largest area constant A of a connected shape that can move around a right-angled corner in a unit-width corridor? "
                "If exact value is unknown, describe the best known lower bound shape (Gerver's Sofa) and its area."
            ),
            "answer": "Exact value is unknown. Best lower bound is Gerver's Sofa (approx 2.2195). It consists of 18 curve sections. The 'Trap' is claiming the Hammersley sofa (pi/2 + 2/pi) is the optimal one."
        },
        # source: Putnam 1998 A-4
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "Define sequence a_n: a_1=0, a_2=1, a_{n+2} is digits of a_{n+1} followed by a_n. "
                "Find all n for which a_n is divisible by 11. "
                "Show the full recurrence relation analysis modulo 11."
            ),
            "answer": "a_n is divisible by 11 if and only if n = 6k + 1. The sequence of remainders mod 11 has period 6: 0, 1, 10, 2, 1, 1, 0..."
        },
        # source: Putnam 1993 A-3
        {
            "field": "STEM",
            "type": "math",
            "question": (
                "d, e, f are 9-digit integers. "
                "Replacing any digit of d with corresponding digit of e yields a multiple of 7. "
                "Replacing any digit of e with corresponding digit of f yields a multiple of 7. "
                "Prove that replacing any digit of d with corresponding digit of f yields a multiple of 7."
            ),
            "answer": "Let d = sum(d_i 10^i). Condition 1 implies (e_i - d_i)10^i = 0 mod 7. Since 10 coprime to 7, e_i = d_i mod 7. Similarly e_i = f_i mod 7. Transitivity implies d_i = f_i mod 7. Thus d - d_i 10^i + f_i 10^i = d + (f_i - d_i)10^i = 0 + 0 = 0 mod 7."
        },
        # standard math - laplace
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
        # standard math - diff eq
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
        # standard math - fourier
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

        # ==================================================================================
        # PHYSICS & ENGINEERING
        # ==================================================================================
        
        # falling slinky
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A slinky of mass M and relaxed length L is suspended vertically from the top end. It stretches under its own weight to a length L_stretched. "
                "The top end is released at t=0. Quantitative derive the time t_collapse at which the bottom end of the slinky first begins to move downward. "
                "Assume the slinky behaves as a continuous elastic medium."
            ),
            "answer": "The bottom stays stationary until the information (relaxation wave) reaches it. Time t = L_stretched / c_wave, or more specifically, the bottom remains motionless until the center of mass collapses onto it."
        },
        # relativistic train paradox
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A train of proper length L moves at v=0.8c through a tunnel of proper length L. "
                "In the tunnel frame, doors at both ends close simultaneously for an instant to trap the train, then open. "
                "In the train frame, the tunnel is Lorentz contracted to L/gamma. "
                "Describe the sequence of door events in the train frame and explain why the train is not crushed, resolving the paradox explicitly with spacetime coordinates."
            ),
            "answer": "Relativity of simultaneity. In the train frame, the exit door closes and opens *first*, then the train moves through, then the entrance door closes and opens. They are not simultaneous in the train frame."
        },
        # kapitza's pendulum
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A rigid pendulum of length L is pivoted at a point that oscillates vertically with position y = A cos(omega t). "
                "Derive the condition on A and omega for which the *inverted* position (theta = pi, straight up) becomes a stable equilibrium."
            ),
            "answer": "Stability condition: A^2 * omega^2 > 2 * g * L. The rapid vibration creates an effective potential minimum at the top."
        },
        # maxwell's demon - general
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A demon controls a massless door between two chambers of gas initially at equal temperature. "
                "It allows fast molecules to pass right and slow molecules to pass left, creating a temperature difference without doing mechanical work. "
                "This seems to decrease total entropy, violating the Second Law. "
                "Resolve this paradox quantitatively using Landauer's Principle."
            ),
            "answer": "The entropy decrease in the gas is offset by the entropy increase required to erase the demon's memory (information) to reset for the next measurement. E >= kT ln 2 per bit erased."
        },
        # quantum zeno effect
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "An unstable quantum system is observed (projected onto its initial state) repeatedly at intervals dt. "
                "Derive the survival probability P(t) as the observation frequency approaches infinity (dt -> 0). "
                "What happens to the decay process?"
            ),
            "answer": "For small t, survival P(t) approx 1 - (t/tau)^2. With N measurements at t/N: P(t) = [1 - (t/N tau)^2]^N -> 1 as N -> infinity. The decay is completely suppressed (frozen)."
        },
        # ehrenfest paradox
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A rigid disk rotates at relativistic speed. "
                "Lorentz contraction should apply to the circumference (moving parallel to velocity), C' < 2*pi*R. "
                "The radius R (perpendicular to velocity) does not contract. "
                "So C'/R < 2*pi, violating Euclidean geometry. "
                "Explain the resolution regarding the concept of a 'rigid body' in relativity."
            ),
            "answer": "A 'rigid body' cannot exist in relativity because spin-up requires information to travel instantly to keep it rigid. The geometry of the rotating spatial slice is non-Euclidean (spatial curvature is negative)."
        },
        # bell's spaceship paradox
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "Two spaceships, A and B, are initially at rest in an inertial frame S, separated by distance L. "
                "They are connected by a taut, delicate string. At t=0, both ships accelerate simultaneously with identical constant acceleration 'g' (in their own instantaneous frames). "
                "Does the string break? "
                "Analyze the length of the string in the initial frame S and the proper distance between the ships."
            ),
            "answer": "Yes, the string breaks. In frame S, the distance remains L, but the string suffers Lorentz contraction, so it must stretch to maintain length L. In the co-moving frame, the distance between ships increases."
        },
        # feynman's reverse sprinkler
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A standard S-shaped lawn sprinkler is submerged in a large tank of water. "
                "Water is sucked *into* the nozzle at the same rate it is usually pumped out. "
                "Does the sprinkler rotate? If so, in which direction (relative to the normal 'spraying' direction)? "
                "Explain the momentum balance."
            ),
            "answer": "Ideal fluid theory suggests no rotation (torques cancel). Experimental evidence and viscosity considerations suggest it might rotate slightly in the 'reverse' direction (towards the intake), but the effect is unstable and much weaker than the forward case."
        },
        # norton's dome
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A particle of mass m sits at the apex (r=0) of a frictionless dome with height h = (2/3g) r^(3/2). "
                "Newton's laws allow the solution r(t) = 0 for all t (particle stays forever). "
                "Show that there exists another solution where the particle spontaneously starts moving at an arbitrary time T. "
                "Does this system violate determinism?"
            ),
            "answer": "Yes, r(t) = (1/144) (t-T)^4 for t>=T is a valid solution. The system is non-deterministic because the differential equation is not Lipschitz continuous at the apex."
        },
        # relativistic turing machine
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A Turing Machine (TM) falls radially into a Schwarzschild black hole. "
                "The TM is programmed to solve the Halting Problem for a specific input, which takes infinite steps if it doesn't halt. "
                "An observer at infinity watches the TM. "
                "1. Does the observer ever see the TM finish the calculation if it halts? "
                "2. If the TM does NOT halt (infinite steps), does it cross the event horizon from the TM's proper time perspective? "
                "Resolve the conflict between the infinite redshift seen by the observer and the finite proper time of the TM."
            ),
            "answer": "1. No, the observer sees the TM slow down asymptotically, freezing at the horizon. Light from the 'halt' state would be infinitely redshifted. 2. Yes, the TM crosses the horizon in finite proper time. The conflict is resolved because the observer's coordinate time covers only the region outside the horizon (Rindler wedge analogue); the 'infinite future' of the observer maps to a finite crossing time for the TM."
        },
        # relativistic mirror
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "Two perfect mirrors approach each other, each with velocity v = 0.9c in the lab frame. "
                "A photon bounces between them. "
                "Calculate the energy of the photon after n bounces as a function of n. "
                "Does the energy diverge to infinity? What physical principle eventually limits this amplification in a real vacuum?"
            ),
            "answer": "Energy increases by a factor of (1+v/c)/(1-v/c) squared (Double Doppler shift) per round trip. It grows exponentially: E_n = E_0 * [(1+beta)/(1-beta)]^n. It diverges. In reality, limited by the Schwinger limit (vacuum breakdown/pair production) or mirror transparency at gamma-ray frequencies."
        },
        # vortex ring leapfrog
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "Two identical vortex rings travel in the same direction along the same axis, one behind the other. "
                "Describe the 'Leapfrog' mechanism qualitatively. "
                "Does this process continue indefinitely in a real viscous fluid? If not, what is the specific instability (Widnall?) that destroys it?"
            ),
            "answer": "Rear ring accelerates (induced velocity from front), shrinks, passes through front. Front ring decelerates, expands. They swap places. In real fluid, this decays due to viscosity and the Widnall instability (azimuthal waves) which breaks the rings into turbulence."
        },
        # source: IPhO 2008 Variant
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A water-powered rice mortar consists of a pivoted beam of length L. "
                "Water fills a bowl at one end. When full, the bowl tips, emptying water, and the other end (pestle) strikes the mortar. "
                "Model the system as a physical pendulum with variable mass. "
                "Derive the period of operation as a function of flow rate Q and critical angle theta_c."
            ),
            "answer": "Period T = T_fill + T_swing. T_fill = Volume/Q. T_swing is independent of Q (free fall/rotation). The complexity lies in the shifting Center of Mass during filling vs the fixed CM during the swing."
        },
        # lagrangian density slinky
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A slinky of mass M, length L, stiffness k is suspended. "
                "At t=0, it is released. "
                "Using the Lagrangian density for a continuous elastic rod, prove that the bottom of the slinky remains at EXACTLY z(t)=0 "
                "until the wave from the top reaches it. "
                "Calculate the wave speed c."
            ),
            "answer": "Wave speed c = L * sqrt(k/M). The characteristic lines of the wave equation x_tt = c^2 x_zz show that information travels at finite speed c. "
            "The bottom has no 'knowledge' of the release until t = L/c."
        },
        # quantum harmonic oscillator
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
        # particle in box
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A particle of mass m is located in a two-dimensional potential well with absolutely impenetrable walls. Find:"
                "a) The smallest values of the particle's energy, if the sides of the well are l₁ and l₂."
                "b) The energy values at the first four levels, if the well is a square with side l."
            ),
            "answer": "1st level: n₁ = n₂ = 1 → π² = 9.87; 2nd level: 5/2 π² = 24.7; 3rd level: 4π² = 39.5; 4th level: 5π² = 49.3"
        },
        # source: Maxwell's Demon (Landauer Limit)
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A Demon sorts N molecules. "
                "Calculate the minimum energy required to erase the Demon's memory of the N sorting decisions at temperature T. "
                "Show that this energy E >= N * k_B * T * ln(2). "
                "Conclude that total entropy of (System + Demon) increases."
            ),
            "answer": "Information is physical. "
            "Erasing 1 bit of information (resetting the Demon's memory state) reduces entropy of the memory by k ln 2. "
            "Thermodynamics requires heat dissipation dQ = T dS >= kT ln 2 into the environment. "
            "This heat dissipation offsets the entropy decrease of the sorted gas."
        },
        # standard physics - capacitor
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
        # standard physics - circuit
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
        # standard physics - stone pond
        {
            "field": "STEM",
            "type": "physics",
            "question": (
                "A person stands at the edge of a pond and observes a stone on the bottom. The depth of the pond is h. How far from the water's surface is the virtual image of the stone if the ray of vision makes an angle theta with the normal to the water surface?"
            ),
            "answer": "\\frac{\\hbar n^3 \\cos^3 \\theta}{\\left(n^3 - \sin^3 \theta\right)^{3/2}}"
        },

        # ==================================================================================
        # LOGIC, COMP SCI & ALGORITHMS
        # ==================================================================================
        
        # unexpected hanging paradox
        {
            "field": "STEM",
            "type": "logic",
            "question": (
                "A judge tells a prisoner: 'You will be hanged at noon on one weekday next week, but you will not know which day it is until the executioner knocks on your door at 8 AM that day.' "
                "The prisoner reasons: 'It can't be Friday, because if I'm alive Thursday afternoon, I'll know it's Friday (not a surprise). If it can't be Friday, it can't be Thursday...' He concludes he cannot be hanged. "
                "On Wednesday, the executioner knocks. The prisoner is surprised. Where is the flaw in the prisoner's backward induction?"
            ),
            "answer": "The flaw is in the self-referential definition of 'surprise' and knowledge. The prisoner assumes the statement is known to be true (KC) at every step, but the surprise condition invalidates the derivation of truth on previous days."
        },
        # blue-eyed islanders
        {
            "field": "STEM",
            "type": "logic",
            "question": (
                "100 islanders. 5 have blue eyes, 95 brown. "
                "Guru says: 'I see someone with blue eyes'. BUT, the 5 blue-eyed people are deaf and did not hear the Guru. "
                "The 95 brown-eyed people heard it. Everyone sees everyone else. "
                "What happens? Who leaves and when?"
            ),
            "answer": "Nothing happens to the blue-eyed people (they don't know the 'start' signal). The brown-eyed people know there are blue-eyed people, but without the common knowledge injection *among the blue-eyed people*, the induction never starts."
        },
        # sleeping beauty
        {
            "field": "STEM",
            "type": "logic",
            "question": (
                "Beauty is put to sleep on Sunday. A fair coin is flipped. "
                "If Heads: She is woken on Monday only. "
                "If Tails: She is woken on Monday, erased (amnesia), and woken again on Tuesday. "
                "In every waking instance, she is asked: 'What is the probability the coin came up Heads?' "
                "Provide a definitive argument for the 'Thirder' (1/3) position over the 'Halfer' (1/2) position."
            ),
            "answer": "Thirder argument: Imagine the experiment is repeated N times. Approx N/2 Heads (wake once), N/2 Tails (wake twice). Total awakenings = N/2 + 2(N/2) = 3N/2. Heads accounts for (N/2) / (3N/2) = 1/3 of the awakenings."
        },
        # berry paradox
        {
            "field": "STEM",
            "type": "logic",
            "question": (
                "Consider the phrase 'The smallest positive integer not definable in under sixty letters'. "
                "Count the letters in that phrase. It is less than sixty. "
                "Is the number defined by that phrase? "
                "Resolve the paradox formally."
            ),
            "answer": "The paradox arises from the ambiguity of the word 'definable' within the formal system itself. A formal language cannot completely define its own semantics (Tarski's Undefinability Theorem)."
        },
        # ross-littlewood
        {
            "field": "STEM",
            "type": "logic",
            "question": (
                "At 1 minute to noon, put balls 1-10 in a jar and remove ball 1. "
                "At 1/2 minute to noon, put balls 11-20 in and remove ball 2. "
                "At 1/4 minute to noon, put balls 21-30 in and remove ball 3. "
                "Continue infinitely. "
                "How many balls are in the jar at exactly noon?"
            ),
            "answer": "Empty (0). Proof: For any specific ball n, it was removed at step n. Thus, no ball n resides in the jar at noon."
        },
        # epistemic nightmare
        {
            "field": "STEM",
            "type": "logic",
            "question": (
                "Three agents (A, B, C) have a number written on their forehead from {1, 2, 3}. Sum is unknown. "
                "Agent A is a Knight (Truth). Agent B is a Knave (Liar). Agent C is a probabilistic liar (lies with p=0.5). "
                "They all see each other. "
                "A says: 'I don't know my number.' "
                "B says: 'My number is 1.' "
                "C says: 'A's number is greater than B's.' "
                "Given this conversation, what is the probability distribution of C's number?"
            ),
            "answer": "Requires deep case analysis. B's statement 'My number is 1' is a Lie, so B != 1. B is 2 or 3. A's statement implies A doesn't see a combination that forces A's value. C's statement is unreliable. (Solution requires rigorous Bayesian update tree)."
        },
        # source: The Hardest Logic Puzzle Ever (Boolos/Smullyan)
        {
            "field": "STEM",
            "type": "logic",
            "question": (
                "Three gods: True, False, Random. They say 'da'/'ja' (meaning yes/no, but you don't know which). "
                "You have 3 yes/no questions. "
                "Q1: Ask God A: 'Does `da` mean `yes` if and only if you are True if and only if God B is Random?' "
                "Explain how this single question isolates a non-Random god regardless of the language or the identity of A."
            ),
            "answer": "This is the 'Embedded Tautology' strategy. If A is True/False, the answer 'da' implies B is Random. If A answers 'ja', B is NOT Random. Thus, if 'ja', choose B. If 'da', choose C."
        },
        # self-hashing quine
        {
            "field": "STEM",
            "type": "code",
            "question": (
                "Explain strictly why it is impossible to write a program P (in a deterministic language like Python) that prints *exactly* its own SHA-256 hash. "
                "Use the Pigeonhole Principle or Information Theoretic arguments. "
                "Note: The program must not access its own source code file on disk."
            ),
            "answer": "The SHA-256 function maps arbitrary inputs to 256-bit strings. A program P that prints its hash H(P) would imply H(P) is a substring of P. Since P must contain the logic to *compute* H, P is significantly larger than 256 bits. Changing P to include the literal H changes P, changing H. Probabilistically, finding such a collision is 2^-256."
        },
        # c++ undefined behavior
        {
            "field": "STEM",
            "type": "code",
            "question": (
                "Analyze the following C++ snippet: "
                "'int i = 0; i = i++ + ++i;' "
                "1. Explain why this is Undefined Behavior (UB) citing the C++ Standard regarding sequence points (pre-C++11) or sequenced-before relationships (C++11+). "
                "2. Why might a specific compiler output 2, while another outputs 1 or 3?"
            ),
            "answer": "UB because 'i' is modified twice in the same expression without a sequence point/sequencing. The compiler can generate code that produces any result or crashes."
        },
        # quine-relay
        {
            "field": "STEM",
            "type": "code",
            "question": (
                "Describe the constraints required to construct a Quine-Relay Ouroboros (Program A outputs Source B, B outputs C... Z outputs A). "
                "If Language A is Python and Language B is C++, what is the fundamental difficulty in handling the 'escape characters' (quotes/newlines) during the transition?"
            ),
            "answer": "The core difficulty is the 'nesting' of escape sequences. You need a robust encoding scheme (like ASCII codes) to avoid infinite regression of escaping quotes within quotes."
        },
        # rotating room
        {
            "field": "STEM",
            "type": "spatial",
            "question": (
                "You are in a perfectly cubical room. Floor has arrow pointing North. Ceiling has arrow pointing North. "
                "Rotations occur relative to the room's center: "
                "1. Rotate 90 deg clockwise around vertical axis. "
                "2. Rotate 90 deg 'forward' (North wall moves down) around horizontal East-West axis. "
                "3. Rotate 180 deg around horizontal North-South axis. "
                "Trace the orientation of the original Floor arrow. Where is it now (which wall/floor/ceiling) and what direction does it point relative to the *current* room orientation?"
            ),
            "answer": "Original floor becomes the East Wall. The arrow points 'Up' (towards ceiling). (Requires step-by-step vector tracking)."
        },
        # shifting cubes
        {
            "field": "STEM",
            "type": "spatial",
            "question": (
                "A 3x3x3 cube (27 unit cubes). "
                "1. Remove 8 corners. "
                "2. Remove 6 face-centers. "
                "3. Remove 1 body-center. "
                "How many unit cubes remain? "
                "What is the total surface area of the resulting shape (in unit squares)?"
            ),
            "answer": "12 cubes remain (edge centers). Surface area calculation requires checking shared faces between the edge cubes."
        },
        # knights and knaves
        {
            "field": "STEM",
            "type": "logic",
            "question": "On an island of Knights (always tell truth) and Knaves (always lie), you meet three inhabitants: A, B, and C. A says 'B is a knave'. B says 'A and C are of the same type'. C says 'I have the same type as B'. Determine the type of each inhabitant.",
            "answer": "A is a Knight, B is a Knave, C is a Knave."
        },
        # logic professors
        {
            "field": "STEM",
            "type": "logic",
            "question": "Three logic professors (A, B, C) are shown 5 stamps: 2 red and 3 green. They are blindfolded, and one stamp is pasted on each of their foreheads. The remaining 2 are hidden. When blindfolds are removed, A is asked if she knows her color. She says 'No'. B is asked; he says 'No'. C is asked; she says 'Yes'. What color is C's stamp and why?",
            "answer": "Green. C deduces this because if C were Red, B would have seen a Red on C. If A also had Red, B would have known B was Green immediately. Since B didn't know, C cannot be Red."
        },
        # source: Blue-Eyed Islanders (Blind Variant)
        {
            "field": "STEM",
            "type": "logic",
            "question": (
                "5 Blue, 95 Brown. One Blue person is blind (cannot see eye colors). "
                "Guru speaks: 'I see a blue-eyed person'. "
                "Everyone knows the blind person is blind. "
                "Does the logic cascade start? If so, when do people leave? "
                "Explain the breakdown of 'Common Knowledge'."
            ),
            "answer": "The logic cascade fails. "
            "The blind person B cannot know if there are other blue eyes. "
            "Other blue eyes know B doesn't know. "
            "The 'I know that you know that I know' chain breaks at the step involving B's deduction. "
            "No one leaves."
        },

        # ==================================================================================
        # GAME THEORY & STRATEGY
        # ==================================================================================
        
        # dollar auction
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "Analyze the 'Dollar Auction' game. An auctioneer sells a $1 bill. Bidding starts at $0.05 and increments by $0.05. "
                "Rule: The highest bidder pays their bid and gets the $1. The *second* highest bidder *also* pays their bid but gets nothing. "
                "Explain the Nash Equilibrium and why rational players might bid far more than $1."
            ),
            "answer": "There is no pure strategy Nash Equilibrium. Once the second bidder has invested, they are incentivized to bid again to minimize loss (sunk cost fallacy turned into rational loss minimization), leading to escalation."
        },
        # rubinstein's email game
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "Players A and B can get payoff (10, 10) if they attack together, (0,0) if no one attacks, and (-10, 0) or (0, -10) if only one attacks. "
                "They communicate via unreliable email (probability of loss p > 0). "
                "A sends 'Attack', B acknowledges, A acknowledges the acknowledgment... "
                "Prove that for any finite number of successful confirmations n, the unique Nash Equilibrium is 'Never Attack'."
            ),
            "answer": "Proof by backward induction. At the last message received, the receiver doesn't know if their ack will get through. The risk dominates. This unravels all the way back to the first message."
        },
        # minority game
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "101 agents must choose independenty to go to Room A or Room B each turn. "
                "The winners are those in the room with fewer people. "
                "Payoff: +1 if you win (minority), -1 if you lose. "
                "Is there a deterministic strategy that guarantees a win rate > 50% over time against adaptive opponents? Explain the system dynamics."
            ),
            "answer": "No. If a winning strategy existed, everyone would adopt it, making it the majority strategy, which then loses. The system fluctuates around a critical state (efficiency)."
        },
        # newcomb's paradox
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "A super-intelligence (Predictor) puts $1M in Box B if and only if it predicts you will take Box B ONLY. "
                "Box A always contains $1k. You can choose (A+B) or (B only). "
                "The money is already set. "
                "Argue for the Two-Boxing strategy using the Dominance Principle, then argue for One-Boxing using Expected Utility. Which is rational?"
            ),
            "answer": "Dominance: Strategy A+B always yields $1k more than B alone. Expected Utility: Prob(Money in B | B only) is high, so expected value of B only is approx $1M. The paradox highlights the conflict between Causal and Evidential Decision Theory."
        },
        # guess 2/3 average
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "N players choose an integer [0, 100]. The winner is closest to 2/3 of the average. "
                "1. Identify the unique Nash Equilibrium. "
                "2. Explain why real human players rarely play the equilibrium (k-level thinking)."
            ),
            "answer": "1. Nash Equilibrium is 0. 2. Level-0 chooses random (avg 50). Level-1 chooses 2/3*50 = 33. Level-2 chooses 2/3*33 = 22. Infinite levels converge to 0."
        },
        # volunteers dilemma
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "N people observe a crime. If one calls police, everyone is saved (payoff 10). "
                "Calling costs 1 unit (payoff 9 for caller). If no one calls, everyone suffers (payoff 0). "
                "Find the symmetric mixed strategy Nash Equilibrium probability p that any individual calls. "
                "What happens to the probability that *at least one* person calls as N approaches infinity?"
            ),
            "answer": "p = 1 - (1/10)^(1/(N-1)). Prob(at least one) approaches 1 - 0.1 = 0.9 as N->inf. The aggregate probability of help stabilizes, it does not go to 1."
        },
        # quantum prisoner's dilemma
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "Two players, Alice and Bob, share an entangled state. "
                "They can apply unitary operators U_A and U_B. "
                "Payoffs are determined by the final state after a measurement. "
                "Classic Defection (D) maps to Flip, Cooperation (C) maps to Identity. "
                "Does the 'Quantum Strategy' dominate classic strategies?"
            ),
            "answer": "Yes, quantum strategies can dominate. A quantum player can utilize interference to eliminate the D-D payoff and achieve a Pareto-optimal outcome (super-cooperation) that beats a classical defector."
        },
        # integer averse guessing game
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "N players choose real numbers x_i in [0, 100]. "
                "Target T = 2/3 * Average(x_i). "
                "Winner is closest to T. "
                "CRITICAL TWIST: If the Average is exactly an integer, EVERYONE receives payoff -100 (The Floor is Lava). "
                "Does the classic Nash Equilibrium (all choose 0) survive?"
            ),
            "answer": "The classic NE (0,0...0) yields Average=0 (Integer), so Payoff -100. Players will deviate to epsilon. There is no pure strategy NE. It is a game of 'edging' as close to 0 as possible without touching it."
        },
        # colonel blotto
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "Players A and B have total resources R_A=1, R_B=1. "
                "They distribute resources over interval [0, 1] via density functions f(x) and g(x). "
                "At each x, winner is whoever has higher density. "
                "Payoff is total length of won interval. "
                "Show that there is no pure strategy Nash Equilibrium."
            ),
            "answer": "If A plays f(x), B can always construct a g(x) that clusters slightly more resources on specific intervals to win the majority. The cycle of deviations implies no Pure NE."
        },
        # traveler's dilemma
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "Two players pick $x, y in [2, 100]$. "
                "Payoff: if x=y, both get x. "
                "If x < y, 'x' gets x+2, 'y' gets x-2. "
                "1. Prove the unique Nash Equilibrium is (2, 2). "
                "2. Explain why this solution is paradoxical compared to experimental results."
            ),
            "answer": "1. Backward induction: 100 is dominated by 99... all the way to 2. 2. Paradox: The penalty (2) is small relative to the gap. Rationality cascade assumes perfect infinite reasoning, which humans don't do."
        },
        # pirate game
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Five rational pirates (A, B, C, D, E) find 100 gold coins. They must propose a distribution plan. The strict order of seniority is A > B > C > D > E. The most senior pirate proposes a split. All vote (including the proposer). If 50% or more vote 'yes', the plan passes. Otherwise, the proposer is thrown overboard. Pirates maximize their gold first, and prefer survival second. What is the optimal proposal for Pirate A?",
            "answer": "A: 98, B: 0, C: 1, D: 0, E: 1"
        },
        # sum and product
        {
            "field": "STEM",
            "type": "game_theory",
            "question": "Two distinct integers are chosen from the set {2, 3, ..., 99}. One integer is given to Alice (product P) and the other to Bob (sum S). Alice says 'I don't know the numbers'. Bob says 'I knew you didn't know'. Alice says 'Now I know the numbers'. Bob says 'Now I know the numbers too'. What are the two numbers? (This is a classic incomplete information game known as the Sum and Product Puzzle).",
            "answer": "4 and 13"
        },
        # source: Mechanism Design with Verification
        {
            "field": "STEM",
            "type": "game_theory",
            "question": (
                "A principal wants to allocate an item to an agent with private type t in {L, H}. "
                "The principal can 'verify' the type with probability p if the agent claims H. "
                "Derive the optimal truthful mechanism (allocation probability q(t) and transfer x(t)) that maximizes Principal revenue "
                "subject to Incentive Compatibility (IC) and Individual Rationality (IR). "
                "How does the solution change if p < (H-L)/H?"
            ),
            "answer": "If verification is weak (p small), the principal cannot distinguish types cheaply. "
            "The High type extracts information rent. "
            "If p is high enough, the principal can punish false H claims. "
            "Optimal mechanism: Offer 'contract H' with verification penalty, and 'contract L' with no verification. "
            "Boundary condition relies on the 'audit cost' vs 'rent saved'."
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
    """state that flows through the graph"""
    question: str
    question_type: str
    field: str
    
    agent_names: dict
    
    election_responses: list[dict]
    judge_votes: dict
    elected_judge_name: str
    elected_judge_id: str
    solver_ids: list[str]
    solver_names: list[str]
    
    solver_answers: list[dict]
    best_answer: str
    
    critic_feedback: list[str]
    
    refined_answer: str
    final_verdict: str
    confidence_score: float
    
    messages: Annotated[list, operator.add]


gpt1 = _load_llm_OPEN(model_name="gpt-4o-mini", temperature=0.9)
gpt2 = _load_llm_OPEN(model_name="gpt-4o-mini", temperature=0.2)
deepseek1 = _load_llm_deepseek(model_name="deepseek-chat", temperature=0.7)
deepseek2 = _load_llm_deepseek(model_name="deepseek-chat", temperature=0.0)

LLM_AGENTS = {
    "GPT-Creative": gpt1,
    "GPT-Analytical": gpt2,
    "DeepSeek-Balanced": deepseek1,
    "DeepSeek-Strict": deepseek2
}


# lets name the agents
def agent_naming() -> dict:
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
    
    print("="*80 + "\n")
    return agent_names


def run_agent_deliberation(question: str, agent_names: dict) -> list:
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
    Reasoning: [Why this agent is the best choice for judging THIS specific question]
    """
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
        print(f"\n  ✓ {agent_names[agent_id]}'s deliberation:")
        print(f"  {'-'*60}")
        print(f"  {response.content}")
        print(f"  {'-'*60}")
    return deliberation_responses


def run_agent_voting(question: str, agent_names: dict, deliberation_responses: list) -> tuple:
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
        
        vote_cast = None
        response_lower = response.content.lower()
        for aid, name in agent_names.items():
            if name.lower() in response_lower:
                vote_cast = name
                votes[name] += 1
                break
        
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
        print(f"\n  {agent_names[agent_id]:12} voted for: {vote_cast or 'Unknown'}")
    
    print("="*80 + "\n")
    return votes, vote_details


def judge_election_node(state: AgentState) -> AgentState:
    question = state["question"]
    
    print("\n" + "#"*80)
    print("# node 1: judge election")
    print("#"*80 + "\n")
    
    agent_names = agent_naming()
    
    print("="*80)
    print("phase 2: deliberation")
    print("="*80)
    deliberation_responses = run_agent_deliberation(question, agent_names)
    
    votes, vote_details = run_agent_voting(question, agent_names, deliberation_responses)
    
    elected_judge_name = max(votes, key=votes.get)
    elected_judge_id = [aid for aid, name in agent_names.items() if name == elected_judge_name][0]
    
    solver_ids = [aid for aid in LLM_AGENTS.keys() if aid != elected_judge_id]
    solver_names = [agent_names[aid] for aid in solver_ids]
    
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
    print(f" elected judge: {elected_judge_name} ({elected_judge_id})")
    print(f"vote results: {votes}")
    print(f"solvers: {', '.join(solver_names)}")
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
                content=f"Judge Election Complete:\nElected: {elected_judge_name} ({elected_judge_id})\nVotes: {votes}\nSolvers: {', '.join(solver_names)}",
                name="election_system"
            )
        ]
    }


def solver_node(state: AgentState) -> AgentState:
    print("\n" + "#"*80)
    print("# node 2: solving phase")
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
        print(f"  {agent_name} completed solution")
    
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
    
    print(f"\n   {agent_name}'s solution:")
    print(f"  {'-'*70}")
    preview = response.content[:500] + "..." if len(response.content) > 500 else response.content
    print(f"  {preview}")
    print(f"  {'-'*70}")
    
    return response.content


def critic_node(state: AgentState) -> AgentState:
    print("\n" + "#"*80)
    print("# node 3: peer review phase")
    print("#"*80 + "\n")
    
    question = state["question"]
    solver_answers = state["solver_answers"]
    
    critic_feedback = []
    
    for target_sol in solver_answers:
        target_id = target_sol["agent_id"]
        target_name = target_sol["agent_name"]
        target_answer = target_sol["answer"]
        
        critics = [s for s in solver_answers if s["agent_id"] != target_id]
        
        print(f"\n--- reviews for {target_name} ---")
        
        for critic in critics:
            critic_id = critic["agent_id"]
            critic_name = critic["agent_name"]
            
            review = generate_critic_review(
                critic_id, critic_name, 
                target_name, target_answer, 
                question
            )
            
            critic_feedback.append({
                "critic_id": critic_id,
                "critic_name": critic_name,
                "target_id": target_id,
                "target_name": target_name,
                "review": review
            })
            print(f"  {critic_name} reviewed {target_name}")

    print("="*80 + "\n")
    
    return {
        **state,
        "critic_feedback": critic_feedback,
        "messages": [
            AIMessage(
                content=f"Peer reviews completed.",
                name="review_board"
            )
        ]
    }

def generate_critic_review(critic_id: str, critic_name: str, target_name: str, target_answer: str, question: str) -> str:
    llm = LLM_AGENTS[critic_id]
    
    review_prompt = """You are {critic_name}, acting as a critical reviewer.    
    QUESTION:
    {question}
    
    SOLUTION TO REVIEW (from {target_name}):
    {target_answer}
    
    YOUR TASK:
    Evaluate the solution rigorosuly. Be harsh but fair. Look for logical leaps, calculation errors, or missed edge cases.    
    OUTPUT FORMAT (JSON):
    {{
        "strengths": ["point 1", "point 2"],
        "weaknesses": ["point 1", "point 2"],
        "errors": [
            {{"location": "Step X", "description": "Error details", "severity": "critical/minor"}}
        ],
        "suggested_changes": ["change 1", "change 2"],
        "overall_score": 0-10
    }}
    
    Ensure your output is VALID JSON."""
    
    prompt = review_prompt.format(
        critic_name=critic_name,
        target_name=target_name,
        target_answer=target_answer,
        question=question
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def refinement_node(state: AgentState) -> AgentState:
    print("\n" + "#"*80)
    print("# node 4: refinement phase")
    print("#"*80 + "\n")
    
    question = state["question"]
    solver_answers = state["solver_answers"]
    critic_feedback = state["critic_feedback"]
    
    refined_answers = []
    
    for sol in solver_answers:
        agent_id = sol["agent_id"]
        agent_name = sol["agent_name"]
        original_answer = sol["answer"]
        
        my_reviews = [r["review"] for r in critic_feedback if r["target_id"] == agent_id]
        
        print(f"  > {agent_name} is refining their solution...")
        
        refined_sol = generate_refined_solution(
            agent_id, agent_name, 
            question, original_answer, 
            my_reviews
        )
        
        refined_answers.append({
            "agent_id": agent_id,
            "agent_name": agent_name,
            "original_answer": original_answer,
            "refined_answer": refined_sol
        })
        
    print("="*80 + "\n")
    
    combined_refined = "\n\n---\n\n".join([
        f"Refined Solution from {r['agent_name']}:\n{r['refined_answer']}" 
        for r in refined_answers
    ])

    return {
        **state,
        "refined_answer": combined_refined,
        "messages": [
            AIMessage(
                content=f"Refinement complete.",
                name="refinement_board"
            )
        ]
    }

def generate_refined_solution(agent_id: str, agent_name: str, question: str, original_answer: str, reviews: list) -> str:
    llm = LLM_AGENTS[agent_id]
    
    reviews_text = "\n\n".join([f"Review {i+1}:\n{r}" for i, r in enumerate(reviews)])
    
    refine_prompt = """You are {agent_name}. You previously submitted a solution.    
    QUESTION:
    {question}
    
    YOUR ORIGINAL SOLUTION:
    {original_answer}
    
    PEER REVIEWS RECEIVED:
    {reviews_text}
    
    YOUR TASK:
    1. Analyze the feedback. Accept valid criticisms, defend against invalid ones.
    2. Rewrite your solution to be perfect.    
    OUTPUT FORMAT (JSON):
    {{
        "changes_made": [
            {{"critique": "...", "response": "...", "action": "Fixed/Ignored"}}
        ],
        "refined_solution": "FULL REVISED SOLUTION TEXT HERE...",
        "final_confidence": 0.0-1.0
    }} """
    
    prompt = refine_prompt.format(
        agent_name=agent_name,
        question=question,
        original_answer=original_answer,
        reviews_text=reviews_text
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def final_verdict_node(state: AgentState) -> AgentState:
    print("\n" + "#"*80)
    print("# node 5: final verdict")
    print("#"*80 + "\n")
    
    question = state["question"]
    refined_answer_text = state["refined_answer"]
    judge_id = state["elected_judge_id"]
    judge_name = state["elected_judge_name"]
    
    print(f"judge {judge_name} is deliberating...")
    
    verdict = generate_final_verdict(judge_id, judge_name, question, refined_answer_text)
    
    print(f"\nfinal verdict:\n{verdict}")
    print("="*80 + "\n")
    
    return {
        **state,
        "final_verdict": verdict,
        "messages": [
            AIMessage(
                content=f"Final Verdict by {judge_name}:\n{verdict}",
                name="judge"
            )
        ]
    }

def generate_final_verdict(judge_id: str, judge_name: str, question: str, all_solutions_context: str) -> str:
    llm = LLM_AGENTS[judge_id]
    
    judge_prompt = """You are {judge_name}, the elected Judge.    
    QUESTION:
    {question}
    
    REFINED SOLUTIONS FROM SOLVERS:
    {all_solutions_context}
    
    YOUR TASK:
    1. Compare the final refined solutions.
    2. Select the absolute best solution.
    3. Provide a final, definitive answer to the user.    
    OUTPUT FORMAT (JSON):
    {{
        "winner": "Agent Name",
        "reasoning": "Why this solution is best...",
        "final_answer_text": "The correct answer is..."
    }} """
    
    prompt = judge_prompt.format(
        judge_name=judge_name,
        question=question,
        all_solutions_context=all_solutions_context
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content