**Hybrid GA–VNS Aircraft Landing Problem Solver**

This repository contains a hybrid metaheuristic solution approach for the Aircraft Landing Problem (ALP).
The proposed method combines Genetic Algorithm (GA) and Variable Neighborhood Search (VNS), with a GRASP-based initialization strategy.

The objective is to determine an optimal landing sequence and corresponding landing times for aircraft, minimizing total penalty costs while satisfying time window and separation constraints.

**Problem description**

The Aircraft Landing Problem consists of scheduling aircraft landings on a single runway.
Each aircraft has:

-  an earliest landing time

-  a target landing time

-  a latest landing time

-  penalty costs for early and late landings

Additionally, a minimum separation time must be respected between consecutive landings, depending on aircraft types.

The goal is to minimize the total weighted deviation from target landing times.

**Methodology**

The solution approach includes the following components:

**GRASP-based initialization**
A greedy randomized construction heuristic generates feasible initial landing sequences.

**Genetic Algorithm**
- Permutation-based encoding
- Tournament selection
- Order crossover (OX)
- Swap mutation
- Elitism strategy

**Variable Neighborhood Search**
Applied to elite solutions using:
- swap neighborhood
- insertion neighborhood
- reversion neighborhood

**Scheduling Landing time**
For each landing sequence, landing times are optimized using a forward pass and backward shifting procedure.

**Repository structure**

-instances/
This folder contains Airland benchmark instance files from OR-Library (e.g., airland1.txt to airland12.txt).

-GA-VNS_aircraft_landing.py
Main Python implementation of the hybrid GA–VNS algorithm.

-README.md
Project description and usage instructions.

**Requirements**

Python 3.x
Standard Python libraries only (random, time, math, etc.)
No external packages are required.

**How to run**

Clone or download the repository.

Make sure the Airland instance files are located in the instances folder.

Open the Python file GA-VNS_aircraft_landing.py.

Set the instance path in the code as follows:

- p, freeze_time, aircraft, S = read_airland_file("instances/airland11.txt")

- You may replace airland11.txt with any other instance file.

To replicate the convergence analysis for Airland 11, uncomment the experimental analysis section at the end of the script.

**Run the script using:**

python GA-VNS_aircraft_landing.py

**Output**

The algorithm returns:

- the best landing sequence found

- the corresponding total penalty cost

**Notes**

The code is written for research and educational purposes.

Computational time may increase significantly for large instances.

Parameters such as population size, number of generations, and GRASP alpha value can be adjusted.
