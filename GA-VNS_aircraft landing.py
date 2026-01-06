# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 11:26:16 2026

@author: DELL
"""

def read_airland_file(filepath):
    """
    Reads an Aircraft Landing Problem (ALP) instance from an OR-Library Airland data file.

    Parameters
    ----------
    filepath : str
        Path to the Airland instance file (e.g., airland11.txt).

    Returns
    -------
    p : int
        Number of aircraft.
    freeze_time : int
        Freeze time parameter of the instance.
    aircraft : list of dict
        List containing aircraft-specific data including time windows and penalty costs.
    separation : list of list
        Separation time matrix between aircraft landings.
    """
    # Read all non-empty lines from the instance file
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    #>Firstt line contains the number of aircraft (p) and the freeze time
    p, freeze_time = map(int, lines[0].split())
    idx = 1
    # List to store aircraft attributes
    aircraft = []
    # Separation time matrix (p x p)
    separation = [[0]*p for _ in range(p)]

    for i in range(p):
        # Read aircraft time windows and penalty parameters
        # a: appearance time, E: earliest, T: target, L: latest
        # g: penalty for early landing, h: penalty for late landing
        a, E, T, L, g, h = map(float, lines[idx].split())
        idx += 1

        aircraft.append({
            "appearance": a,
            "earliest": E,
            "target": T,
            "latest": L,
            "g": g,
            "h": h
        })

        # Read separation times required after aircraft i lands
        sep_times = []
        while len(sep_times) < p:
          sep_times.extend(map(int, lines[idx].split()))
          idx += 1
        # Fill separation matrix for aircraft i
        for j in range(p):
            separation[i][j] = sep_times[j]

    return p, freeze_time, aircraft, separation
# Load Airland benchmark instance
p, freeze_time, aircraft, S = read_airland_file(r"C:\Users\DELL\Desktop\airland11.txt")

#initial population oluşturma: perm.based
import random

def initialize_population(pop_size, p, aircraft, S, alpha):
    """
    Generates an initial population of landing sequences using a GRASP-based
    greedy randomized construction heuristic.

    Parameters
    ----------
    pop_size : int
        Number of individuals (landing sequences) in the population.
    p : int
        Number of aircraft.
    aircraft : list of dict
        Aircraft data including time windows and penalty parameters.
    S : list of list
        Separation time matrix between aircraft.
    alpha : float
        GRASP parameter controlling the greediness–randomness balance
        (alpha = 1: greedy, alpha = 0: fully random).

    Returns
    -------
    population : list of list
        A list of feasible landing sequences (permutations of aircraft indices).
    """
    population = []
    # Generate pop_size individuals
    for _ in range(pop_size):
        individual = []
        # Initially, all aircraft are candidates
        candidates = list(range(p))
        # Track last scheduled landing to respect separation constraints
        last_landing_time = 0
        last_aircraft = None
        # Construct one landing sequence incrementally
        while candidates:
            
            costs = []  
            # Evaluate each candidate aircraft
            for c in candidates:
                # Earliest feasible landing time
                earliest_possible = aircraft[c]['earliest']
                # Enforce separation constraint with the previously scheduled aircraft
                if last_aircraft is not None:
                    gap_req = last_landing_time + S[last_aircraft][c]
                    earliest_possible = max(earliest_possible, gap_req)
                
                # Compute penalty cost based on deviation from target time
                target = aircraft[c]['target']
                if earliest_possible < target:
                    cost = (target - earliest_possible) * aircraft[c]['g']
                else:
                    cost = (earliest_possible - target) * aircraft[c]['h']
                
                # Strongly penalize infeasible landings beyond latest time
                if earliest_possible > aircraft[c]['latest']:
                    cost += 1e24
                
                costs.append((cost, c, earliest_possible))
            
            # Determine Restricted Candidate List (RCL)
            c_min = min(costs, key=lambda x: x[0])[0]
            c_max = max(costs, key=lambda x: x[0])[0]
            
            # GRASP threshold rule
            threshold = c_min + alpha* (c_max - c_min)
            rcl = [item for item in costs if item[0] <= threshold]
            
            # Randomly select one aircraft from the RCL
            chosen_cost, chosen_aircraft, chosen_time = random.choice(rcl)
            # Update sequence and state
            individual.append(chosen_aircraft)
            candidates.remove(chosen_aircraft)
            last_landing_time = chosen_time
            last_aircraft = chosen_aircraft
            
        population.append(individual)
        
    return population

# Population size
pop_size = 150

# Initialize population using GRASP-based construction
population = initialize_population(pop_size, p, aircraft, S, alpha=0.5)



def schedule_landing_times(sequence, p, aircraft, S):
    """
Given a landing sequence, computes feasible landing times and the
corresponding total penalty cost using a forward–backward adjustment
procedure.

The procedure consists of:
(i) a forward pass to ensure feasibility with respect to earliest
    landing times and separation constraints,
(ii) a backward shifting phase to move landings closer to target times
     without violating feasibility,
(iii) evaluation of the total penalty cost.

Parameters
----------
sequence : list
    A permutation of aircraft indices representing the landing order.
p : int
    Number of aircraft.
aircraft : list of dict
    Aircraft-specific data including time windows and penalty parameters.
S : list of list
    Separation time matrix.

Returns
-------
t : list
    Optimized landing times for each aircraft.
optimized_cost : float
    Total penalty cost of the optimized landing schedule.
    """
    # -------------------------------
    # 1. Forward pass: feasibility
    # -------------------------------
    # Initialize landing times
    t = [0.0] * p
    for i in range(p):
        curr = sequence[i]
        
        # Start with earliest allowable landing time
        t[curr] = aircraft[curr]['earliest']
        
        # Enforce separation constraint with previous aircraft
        if i > 0:
            prev = sequence[i-1]
            t[curr] = max(t[curr], t[prev] + S[prev][curr])

    # ----------------------------------------
    # 2. Backward shifting toward target times
    # ----------------------------------------
    # Move backward through the sequence and shift aircraft
    # closer to their target times without violating constraints
    for i in range(p-1, -1, -1):
        curr = sequence[i]
        target = aircraft[curr]['target']
        
        # Upper bound for shifting (latest time and separation constraint)
        upper_limit = aircraft[curr]['latest']
        if i < p - 1:
            nxt = sequence[i+1]
            upper_limit = min(upper_limit, t[nxt] - S[curr][nxt])
            
        # Shift only if it moves the aircraft closer to its target
        if target > t[curr]:
            t[curr] = min(target, upper_limit)
            
    # -------------------------------
    # 3. Penalty cost computation
    # -------------------------------
    optimized_cost = 0.0
    for ac_id in sequence:
        target = aircraft[ac_id]['target']
        dist = t[ac_id] - target
        
        # Heavy penalty for infeasible solutions (latest time violation)
        if t[ac_id] > aircraft[ac_id]['latest']: optimized_cost += 1000000 + (t[ac_id] - aircraft[ac_id]['latest']) * 10000
        
        # Early landing penalty
        elif dist < 0: optimized_cost += abs(dist) * aircraft[ac_id]['g']
        
        # Late landing penalty
        else: optimized_cost += dist * aircraft[ac_id]['h']
        
    return t, optimized_cost


def tournament_selection(pop, fitness_scores, k=3 ):
    """
    Selects one individual from the population using tournament selection.

    Parameters
    ----------
    pop : list
        Current population of landing sequences.
    fitness_scores : list
        Corresponding fitness (penalty cost) values.
    k : int, optional
        Tournament size (default is 3).

    Returns
    -------
    selected : list
        The selected individual with the best fitness in the tournament.
    """
    
    # Randomly select k individuals
    selected_indices = random.sample(range(len(pop)), k)
    
    # Choose the individual with the minimum penalty cost
    best_idx = min(selected_indices, key=lambda idx: fitness_scores[idx])
    
    return pop[best_idx]

#çaprazlama - OX Crossover
def order_crossover(parent1, parent2,pc):
    """
    Performs Order Crossover (OX) to generate a child solution
    while preserving the permutation structure.

    Parameters
    ----------
    parent1, parent2 : list
        Parent landing sequences.
    pc : float
        Crossover probability.

    Returns
    -------
    child : list
        Offspring landing sequence.
    """
    
    # No crossover with probability (1 - pc)
    if random.random() > pc:
        return list(parent1)
    
    p = len(parent1)
    
    # Randomly select two cut points
    a,b = sorted(random.sample(range(p), 2))
    
    child1 = [None]*p
    
    # Copy the segment from the first parent
    child1[a:b] = parent1[a:b]
    
    #boş kalanlar parent 2 den gelecke.
    # Fill remaining positions from the second parent
    p2_remaining =[item for item in parent2 if item not in child1]
    idx=0
    for i in range(p):
        if child1[i] is None:
            child1[i] = p2_remaining[idx]
            idx +=1
    return child1

def mutate(individual, pm):
    """
    Applies swap mutation to a landing sequence.

    Parameters
    ----------
    individual : list
        Landing sequence.
    pm : float
        Mutation probability.

    Returns
    -------
    individual : list
        Mutated landing sequence.
    """
    if random.random() < pm:
        # Swap two randomly selected aircraft
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def vns_refinement(sequence, p, aircraft, S, max_attempts=20):
    """
    Improves a landing sequence using Variable Neighborhood Search (VNS).

    Three neighborhood structures are explored:
    1) Swap
    2) Insertion
    3) Reversion

    Parameters
    ----------
    sequence : list
        Initial landing sequence.
    p : int
        Number of aircraft.
    aircraft : list of dict
        Aircraft data.
    S : list of list
        Separation time matrix.
    max_attempts : int, optional
        Number of attempts per neighborhood.

    Returns
    -------
    best_seq : list
        Locally improved landing sequence.
    """
    best_seq = list(sequence)
    
    _, best_fit = schedule_landing_times(best_seq, p, aircraft, S)
    
    # Neighborhood 1: Swap
    for _ in range(max_attempts):
        i, j = random.sample(range(p), 2)
        test_seq = list(best_seq)
        test_seq[i], test_seq[j] = test_seq[j], test_seq[i]
        # ÖNEMLİ: Maliyeti optimize ederek ölçüyoruz
        _, test_fit = schedule_landing_times(test_seq, p, aircraft, S)
        if test_fit < best_fit:
            best_fit = test_fit
            best_seq = test_seq

    # Neighborhood 2: Insertion
    for _ in range(max_attempts):
        idx = random.randint(0, p-1)
        test_seq = list(best_seq)
        ac = test_seq.pop(idx)
        new_pos = random.randint(0, p-1) 
        test_seq.insert(new_pos, ac)
        
        _, test_fit = schedule_landing_times(test_seq, p, aircraft, S)
        if test_fit < best_fit:
            best_fit = test_fit
            best_seq = test_seq
            
    # Neighborhood 3: Reversion
    for _ in range(10):
        a, b = sorted(random.sample(range(p), 2))
        if b - a > 1:
            test_seq = list(best_seq)
            test_seq[a:b] = reversed(test_seq[a:b])
            _, test_fit = schedule_landing_times(test_seq, p, aircraft, S)
            if test_fit < best_fit:
                best_fit = test_fit
                best_seq = test_seq
    return best_seq

def solve_alp(p, aircraft, S, pop_size=150, max_gen=5000, alpha=0.5, pc_val=0.9):
    """
    Solves the Aircraft Landing Problem (ALP) using a hybrid
    Genetic Algorithm (GA) and Variable Neighborhood Search (VNS) framework.

    The algorithm integrates:
    - GRASP-based population initialization,
    - GA-based global exploration,
    - VNS-based local refinement,
    - adaptive crossover and mutation control,
    - early stopping based on stagnation.

    Parameters
    ----------
    p : int
        Number of aircraft.
    aircraft : list of dict
        Aircraft data including time windows and penalty parameters.
    S : list of list
        Separation time matrix.
    pop_size : int, optional
        Population size (default is 150).
    max_gen : int, optional
        Maximum number of generations.
    alpha : float, optional
        GRASP parameter for initial population construction.
    pc_val : float, optional
        Base crossover probability.

    Returns
    -------
    best_overall_seq : list
        Best landing sequence found.
    best_overall_cost : float
        Penalty cost of the best solution.
    """
    
    # Initialize population using GRASP
    pop = initialize_population(pop_size, p, aircraft, S, alpha=alpha)
    best_overall_cost = float('inf')
    best_overall_seq = None
    stagnation_counter = 0

    no_improvement_limit = 500  
    early_stop_counter = 0
    
    for gen in range(max_gen):
        # -------------------------
        # Fitness evaluation
        # -------------------------
        fitness_scores = []
        for ind in pop:
            _, cost = schedule_landing_times(ind, p, aircraft, S)
            fitness_scores.append(cost)
        
        sorted_indices = sorted(range(len(pop)), key=lambda k: fitness_scores[k])
        current_best_idx = sorted_indices[0]
        
        # -------------------------
        # Best solution update
        # -------------------------
        if fitness_scores[current_best_idx] < best_overall_cost:
            best_overall_cost = fitness_scores[current_best_idx]
            best_overall_seq = list(pop[current_best_idx])
            stagnation_counter = 0
            early_stop_counter = 0
        else:
            stagnation_counter += 1
            early_stop_counter += 1
        
        # Early stopping criterion
        if early_stop_counter >= no_improvement_limit:
            break
            
        # -------------------------
        # Adaptive parameters
        # -------------------------
        if stagnation_counter > 50:
            pc = 0.6
            pm = 0.1
            
            # Partial population shuffle to increase diversity
            for i in range(1, int(pop_size * 0.5)):
                random.shuffle(pop[i]) 
            stagnation_counter = 30 
        else:
            pc = pc_val
            pm = 0.01
        
        # -------------------------
        # Elitism + VNS refinement
        # -------------------------
        num_elites = max(1, int(pop_size * 0.03))
        new_gen = [] 
        for i in range(num_elites):
            elite_idx = sorted_indices[i]
           
            refined_elite = vns_refinement(pop[elite_idx], p, aircraft, S, max_attempts=20)
            new_gen.append(refined_elite)
            
            
            _, refined_cost = schedule_landing_times(refined_elite, p, aircraft, S)
            
            if refined_cost < best_overall_cost:
                best_overall_cost = refined_cost
                best_overall_seq = list(refined_elite)
                early_stop_counter = 0

        # -------------------------
        # GA reproduction
        # -------------------------
        while len(new_gen) < pop_size:
            p1 = tournament_selection(pop, fitness_scores)
            p2 = tournament_selection(pop, fitness_scores)
            
            child = order_crossover(p1, p2, pc) 
            child = mutate(child, pm) 
            new_gen.append(child)
            
        pop = new_gen

    return best_overall_seq, best_overall_cost



#optional
# def display_detailed_schedule(best_sequence, p, aircraft, S):
#     """
#     Displays the optimized landing schedule in a tabular format.
#     This function is used only for reporting and visualization purposes.
#     """
    
#     times, _ = schedule_landing_times(best_sequence, p, aircraft, S)
       
#     for i in range(p):
#         curr = best_sequence[i]
#         target = aircraft[curr]['target']
#         scheduled_time = times[curr]
#         delay = scheduled_time - target
        
# best_seq, best_cost = solve_alp(p, aircraft, S, pop_size=150, max_gen=5000,alpha=0.5)
# display_detailed_schedule(best_seq, p, aircraft, S)
# print(f"\nFinal Maliyet: {best_cost}")





# ------------------------------------------------------------
# Experimental analysis: effect of max_gen on solution quality
# ------------------------------------------------------------
# import numpy as np
# import time
# gen_test_values = [200,500, 1000, 1500, 3000, 5000]
# BKS = 12418.32  # Airland 11'in Best Known Solution 

# results_for_report = []

# for m_gen in gen_test_values:
#     run_costs = []
#     start_time = time.time()
    
#     # 5'er koşu (vakit kazanmak için)
#     for _ in range(2):
#         _, cost = solve_alp(p, aircraft, S, pop_size=150, max_gen=m_gen, alpha=0.5)
#         run_costs.append(cost)
    
#     avg_cost = np.mean(run_costs)
#     best_cost = np.min(run_costs)
    
#     gap = ((best_cost - BKS) / BKS) * 100
    
    
#     results_for_report.append({
#         "MaxGen": m_gen,
#         "Best": best_cost,
#         "Gap": gap,
       
#     })
#     print(f"MaxGen {m_gen} | Gap: %{gap:.2f}")




