# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:13:40 2026

@author: trist
"""

import os

#Forces the following libraries to use only on thread
#Avoid competition between parellel processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import math
import csv
from numba import njit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from check_params import get_filename

@njit()
def get_fitness(x: int, y: int, lattice: list[list[int]]) -> float:
    """
    Compute the fitness of a bacteria in the lattice.
    Its fitness will dictate the outcome of reproduction and selection events.

    Parameters
    ----------
    x : int
        Row of the bacteria on the lattice.
    y : int
        Column of the bacteria on the lattice.
    lattice : list[list[int]]
        Spatial distribution of the bacteria.

    Returns
    -------
    float
        Fitness of the bacteria at position [x, y] on the lattice.

    """
    assert lattice[x, y] not in (4, 5)       #Fitness is not relevant for empty space and insulators
    #Count the number of bacteria of each genotype in the neighbourhood
    neighbours = [0, 0, 0, 0, 0, 0]
    for i in range(-RADIUS, RADIUS+1):
        for j in range(-RADIUS, RADIUS+1):
            neighbours[lattice[(x+i)%L, (y+j)%L]] += 1
    #Deduce the fitness gain from each secretion
    arg1 = sum([n*g for n,g in zip(neighbours, GOOD1_PER_CELL)])
    arg2 = sum([n*g for n,g in zip(neighbours, GOOD2_PER_CELL)])
    fitness = BASAL_FITNESS + WEIGHT*max(arg1, arg2) + (1-WEIGHT)*min(arg1, arg2) - COST_PER_CELL[lattice[x, y]]
    return max(0., fitness)


@njit()
def all_empty(lattice: list[list[float]]) -> bool:
    """
    Check if there are still alive cells on the lattice.
    This is to prevent infinite while loops when chosing a random bacteria on the lattice.

    Parameters
    ----------
    lattice : list[list[float]]
        Spatial distribution of the bacteria.

    Returns
    -------
    bool
        True if and only if all space on the lattice is not occupied.

    """
    for i in range(L):
        for j in range(L):
            if lattice[i, j] != 4:
                return False
    return True


@njit()
def reaction_single(lattice: list[list[int]]) -> tuple[int]:
    """
    Chose a reaction of reproduction, selection or death and carries its effect.

    Parameters
    ----------
    lattice : list[list[int]]
        Spatial distribution of the bacteria.

    Returns
    -------
    tuple[int]
        Record of the effect of the reaction.

    """
    #Fail if all bacteria are dead
    if all_empty(lattice):
        return (-1, -1, -1, -1, -1, -1, -1)
    #Otherwise chose a random bacteria
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)
    while lattice[x, y] == 4:
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
    #Chose a random adjacent space
    d = np.random.randint(0, 4)
    if d == 0:   dx, dy = 0, 1
    elif d == 1: dx, dy = 1, 0
    elif d == 2: dx, dy = -1, 0
    else:        dx, dy = 0, -1
    nx, ny = (x+dx)%L, (y+dy)%L
    #Chose randomly the reaction type
    reaction_choice = np.random.rand()
    SUM_RATES = SELECTION_RATE + REPRODUCTION_RATE + DEATH_RATE
    if reaction_choice < REPRODUCTION_RATE / SUM_RATES:
        #Reproduction event
        if lattice[nx, ny] == 4:
            if lattice[x, y] != 5:
                f = get_fitness(x, y, lattice)
                if np.random.rand() < f:
                    lattice[nx, ny] = lattice[x, y]
                    return (0, lattice[x, y], nx, ny, x, y, -1)
            else:
                #Fitness is irrelevant for insulators, so they instead
                #reproduce with a constant low rate
                if np.random.rand() < .1:
                    lattice[nx, ny] = 5
                    return (0, 5, nx, ny, x, y, -1)
    elif reaction_choice < (REPRODUCTION_RATE + DEATH_RATE) / SUM_RATES:
        #Death event
        genotype = lattice[x, y]
        lattice[x, y] = 4
        return (2, genotype, x, y, -1, -1, -1)
    else:
        #Selection event
        if (lattice[x, y] != lattice[nx, ny]                                   #No selection between cells of the same genotype
            and lattice[x, y] not in (4, 5)                                    #No selection with empty space and insulators
            and lattice[nx, ny] not in (4, 5)                                  #No selection with empty space and insulators
            and not (lattice[x, y] in (1, 2) and lattice[nx, ny] in (1, 2))):  #No selection between cross-feeders
            fn = get_fitness(nx, ny, lattice)
            f = get_fitness(x, y, lattice)
            relative_fitness = fn - f
            #Kill the bacteria if its fitness is lower than its neighbour
            if relative_fitness > 0 or (relative_fitness == 0 and np.random.rand() < .5):
                genotype = lattice[x, y]
                lattice[x, y] = 4
                return (1, genotype, x, y, lattice[nx, ny], nx, ny)
    return (-1, -1, -1, -1, -1, -1, -1) #Failed reaction
            

@njit()
def exchange(lattice: list[list[int]], exchangeRate: float) -> tuple[int]:
    """
    Carry out (or not) one exchange between two adjacent bacteria.

    Parameters
    ----------
    lattice : 
        Spatial distribution of the bacteria.
    exchangeRate : float
        Indicates the average number of exchange events initiated by each 
        bacteria during one generation.

    Returns
    -------
    tuple[int]
        Record of the outcome of the exchange event.
    """
    #Fail if all bacteria are dead
    if all_empty(lattice):
        return (-1, -1, -1, -1, -1, -1, -1)
    #Otherwise, chose a random bacteria
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)
    while lattice[x, y] == 4:
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
    directions = [[x, (y+1)%L], [x, (y-1)%L], [(x+1)%L, y], [(x-1)%L, y]]
    #Check if the exchange event will fail for all directions (for efficiency)
    check = True
    if BLOCKING:
        for d in directions:
            if lattice[d[0], d[1]] == 4:
                check = False
        if check:
            return (-1, -1, -1, -1, -1, -1, -1)  #Blocked from all directions
    check = True
    for d in directions:
        if lattice[d[0], d[1]] != lattice[x, y]:
            check = False
    if check:
        return (-1, -1, -1, -1, -1, -1, -1)      #Exchange will be trivial
    #Choose the direction where the bacteria will move
    #(i.e. the adjacent space with which it will swap places)
    if CHEMOTAXIS != 0 and lattice[x, y] != 5:
        #If chemotaxis is on, select how much each cell will attract the bacteria
        if abs(CHEMOTAXIS) == 1: #Chemotaxis towards what you do not produce
            attraction_list = [[0., 1., 1., 1., 0., 0.],
                               [0., 0., 1., 1., 0., 0.],
                               [0., 1., 0., 1., 0., 0.],
                               [0., 0., 0., 0., 0., 0.]]
            attraction = attraction_list[lattice[x, y]]
        elif abs(CHEMOTAXIS) == 2: #Chemotaxis towards what you produce
            attraction_list = [[0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 1., 0., 0.],
                               [0., 0., 1., 1., 0., 0.],
                               [0., 1., 1., 1., 0., 0.]]
            attraction = attraction_list[lattice[x, y]]
        elif abs(CHEMOTAXIS) == 3: #Chemotaxis towards all secretions
            attraction = [0., 1., 1., 1., 0., 0.]
        weights = [0, 0, 0, 0]
        for di, d in enumerate(directions):
            #Count the number of cells of each genotype in each direction
            neighbours = [0, 0, 0, 0, 0, 0]
            for i in range(-DETECTION_RADIUS, DETECTION_RADIUS+1):
                for j in range(-DETECTION_RADIUS, DETECTION_RADIUS+1):
                    neighbours[lattice[(d[0]+i)%L, (d[1]+j)%L]] += 1
            #Dot product with the attraction to each genotype to obtain the "weight" of the direction
            weights[di] = sum([n*a for n,a in zip(neighbours, attraction)])
        if CHEMOTAXIS < 0:
            #Non-linear scaling of weights
            weights = [np.exp(BETA*w) for w in weights]
        #Normalisation
        total_weight = sum(weights)
        if total_weight > 0:
            probas = [w/total_weight for w in weights]
            #Randomly chose the direction, according to the weight of each direction
            pick = np.random.rand()
            if pick < probas[0]:
                di = 0 #up
            elif pick < probas[0] + probas[1]:
                di = 1 #down
            elif pick < probas[0] + probas[1] + probas[2]:
                di = 2 #right
            else:
                di = 3 #left
        else:
            di = np.random.randint(0, 4)
    else:
        #With unbiased motility, the chosen direction is purely random
        di = np.random.randint(0, 4)
    #Get the coordinates of the adjacent cell which has been selected
    if di == 0:   dx, dy = 0, 1
    elif di == 1: dx, dy = 0, -1
    elif di == 2: dx, dy = 1, 0
    else:         dx, dy = -1, 0
    nx, ny = (x+dx)%L, (y+dy)%L
    #Check if the exchange fails
    if (lattice[x, y] == lattice[nx, ny]         #Trivial exchange
        or lattice[nx, ny] != 4 and BLOCKING):   #Blocked exchange
        return (-1, -1, -1, -1, -1, -1, -1)
    if MIN_EXCHANGE_RATE is not None:
        #Count the number of cells in the neighbourhood
        count = 0
        for i in range(-DETECTION_RADIUS, DETECTION_RADIUS+1):
            for j in range(-DETECTION_RADIUS, DETECTION_RADIUS+1):
                if lattice[(x+i)%L, (y+j)%L] != 4:
                    count += 1
        #Deduce the local cell density
        density = count / (DETECTION_RADIUS*2 + 1)**2
        if density > DD_THRESHOLD:
            #Cell density too high -> the minimum motility rate is applied
            ratio_epsilons = MIN_EXCHANGE_RATE / exchangeRate
            if np.random.random() > ratio_epsilons:
                return (-1, -1, -1, -1, -1, -1, -1)
    #Carry out the exchange if it has not failed
    lattice[x, y], lattice[nx, ny] = lattice[nx, ny], lattice[x, y]
    return (3, lattice[nx, ny], x, y, lattice[x, y], nx, ny)
            

@njit()
def simulation_one_generation(lattice: list[list[int]], exchangeRate: float) -> list[tuple[int]]:
    """
    Carry out all the reactions and exchanges happening in one generation.

    Parameters
    ----------
    lattice : list[list[int]]
        Spatial distribution of the bacteria.
    exchangeRate : float
        Indicates the average number of exchange events initiated by each 
        bacteria during one generation.

    Returns
    -------
    list[tuple[int]]
        Record of all of the reactions and exchanges that happened during the generation.

    """
    logGen = [(-1, -1, -1, -1, -1, -1, -1)]
    SUM_RATES = SELECTION_RATE + REPRODUCTION_RATE + DEATH_RATE
    #A generation is defined as time so that each cell has reacted on average once
    nb_alive_cells = len(np.argwhere(lattice != 4))
    nb_reactions = math.floor(nb_alive_cells * SUM_RATES)
    for _ in range(nb_reactions):
        #Compute the number of exchanges that will happen before one reaction
        #(using a geometric random variable is faster than mixing exchange and reactions when exchangeRate is high)
        if exchangeRate != 0:
            nb_exchanges = math.floor(math.log(np.random.rand()) / math.log(exchangeRate/(SUM_RATES+exchangeRate)))
            for __ in range(nb_exchanges):
                if SAVE_LOG:
                    logGen.append(exchange(lattice, exchangeRate))
                else:
                    exchange(lattice, exchangeRate)
        #Carry out one reaction
        if SAVE_LOG:
            logGen.append(reaction_single(lattice))
        else:
            reaction_single(lattice)
    return logGen
    

@njit()
def count_cells_in_lattice(lattice: list[list[int]]) -> list[int]:
    """
    Count the number of cells of each genotype in the lattice.

    Parameters
    ----------
    lattice : list[list[int]]
        Spatial distribution of the bacteria.

    Returns
    -------
    count : list[int]
        Number of cells of each genotype on the lattice.

    """
    count = [0,0,0,0,0,0]
    for row in lattice:
        for cell in row:
            count[cell] += 1
    return count



def simulation_global(exchangeRate: float, seed: int):
    np.random.seed(seed)
    numbers = range(INITIAL_SPARSITY)
    lattice = np.random.choice(numbers, (L, L))
    if INSULATORS_DENSITY > 0:
        for i in range(5, 5+INSULATORS_DENSITY):
            lattice = np.where(lattice == i, 5, lattice)
        lattice = np.where(lattice > 5, 4, lattice)
    else:
        lattice = np.where(lattice > 4, 4, lattice)
    if not SPAWN_FULL_PRODUCERS:
        lattice = np.where(lattice == 3, 4, lattice)
    if not SPAWN_CROSS_FEEDERS:
        lattice = np.where(lattice == 1, 4, lattice)
        lattice = np.where(lattice == 2, 4, lattice)
    if not SPAWN_FULL_DEFECTORS:
        lattice = np.where(lattice == 0, 4, lattice)
    lattice = lattice.astype(np.int8)
    history, log = [], []
    for i in range(MAX_GEN):
        if SAVE_LOG:
            logGen = simulation_one_generation(lattice, exchangeRate)
            log.append([l for l in logGen if l != (-1, -1, -1, -1, -1, -1, -1)])
        else:
            simulation_one_generation(lattice, exchangeRate)
        if SAVE_HISTORY:
            history.append(np.copy(lattice))
        cell_count = count_cells_in_lattice(lattice)
        nb_extinct_strands = 0
        if cell_count[0] == 0:
            nb_extinct_strands += 1
        if cell_count[1] == 0 and cell_count[2] == 0:
            nb_extinct_strands += 1
        if cell_count[3] == 0:
            nb_extinct_strands += 1
        if nb_extinct_strands >= 2:
            break
    last_gen = i
    final_density = len(lattice[lattice < 4])
    extinctions = [int(i not in lattice) for i in range(4)]
    last_strand = extinctions.index(0) if 0 in extinctions else -1
    if not SAVE_HISTORY:
        for i in range(L):
            simulation_one_generation(lattice, exchangeRate)
            density2 = len(lattice[lattice < 4])
            if density2 < L/4:
                out = ([1, 1, 1, 1], final_density, last_gen, last_strand)
            else:
                out = (extinctions, final_density, last_gen, last_strand)
    else:
        if SAVE_LOG:
            out = (history, log)
        else:
            out = history
        print(extinctions, final_density, last_gen, last_strand) 
    return out


def submit_one(executor, exchangeRate):
    seed = np.random.SeedSequence().spawn(1)[0]
    return executor.submit(simulation_global, exchangeRate, int(seed.generate_state(1)[0]))


def get_IC95(extinctions):
    tries = len(extinctions)
    if tries == 0:
        return np.array([9999, 9999, 9999, 9999])
    std = np.std(extinctions, axis=0)
    SEM = std/tries**.5
    IC95 = 1.96*SEM
    return IC95


def save_results(filenameOut, extinctions, densities, last_gens, lasts):
    with open(r'./data/' + filenameOut, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for ext, dens, lgen, last in zip(extinctions, densities, last_gens, lasts):
            writer.writerow(list(ext) + [dens] + [lgen] + [last])
    print(f'Saved {len(extinctions)} results.')
    

def make_seed():
    return int(np.random.SeedSequence().spawn(1)[0].generate_state(1)[0])
            

exchanges = np.arange(0., 20.5, .5)
IC95_THRESHOLD = .05
MIN_REPETITIONS = 50
MAX_GEN = 999999999999
CURRENT_PARAMS_PATH = r'./current_params.json'
DATA_FOLDER = r'./data/'

with open(CURRENT_PARAMS_PATH, 'r') as f:
    PARAMS = json.load(f)
    
SAVE_HISTORY, SAVE_LOG = PARAMS['SAVE_HISTORY'], PARAMS['SAVE_LOG']
GOOD, COST = PARAMS['GOOD'], PARAMS['COST']
L = PARAMS['L']
SELECTION_RATE = PARAMS['SELECTION_RATE']
REPRODUCTION_RATE = PARAMS['REPRODUCTION_RATE']
DEATH_RATE = PARAMS['DEATH_RATE']
BASAL_FITNESS, WEIGHT = PARAMS['BASAL_FITNESS'], PARAMS['WEIGHT']
INITIAL_SPARSITY = PARAMS['INITIAL_SPARSITY']
RADIUS, DETECTION_RADIUS = PARAMS['RADIUS'], PARAMS['DETECTION_RADIUS']
CHEMOTAXIS, BETA = PARAMS['CHEMOTAXIS'], PARAMS['BETA']
SPAWN_FULL_PRODUCERS = PARAMS['SPAWN_FULL_PRODUCERS']
SPAWN_CROSS_FEEDERS = PARAMS['SPAWN_CROSS_FEEDERS']
SPAWN_FULL_DEFECTORS = PARAMS['SPAWN_FULL_DEFECTORS']
INSULATORS_DENSITY = PARAMS['INSULATORS_DENSITY']
MIN_EXCHANGE_RATE, DD_THRESHOLD = PARAMS['MIN_EXCHANGE_RATE'], PARAMS['DD_THRESHOLD']
BLOCKING = PARAMS['BLOCKING']


GOOD1_PER_CELL = (0., GOOD, 0., GOOD, 0., 0.) 
GOOD2_PER_CELL = (0., 0., GOOD, GOOD, 0., 0.)
COST_PER_CELL = (0., COST, COST, 2*COST, 0., 0.)

if __name__ == '__main__':
    n_processors = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=n_processors) as executor:
        if SAVE_HISTORY:
            #exchangeRate = round(float(input('Exchange rate ? ')), 2)
            exchangeRate = 5.
            seed = np.random.SeedSequence().spawn(1)
            if SAVE_LOG:
                filenameHistory, filenameLog = get_filename(exchangeRate, PARAMS)
                history, log = simulation_global(exchangeRate, int(seed[0].generate_state(1)[0]))
                np.savez_compressed('./data/' + filenameHistory, *[h for h in history])
                np.savez_compressed('./data/' + filenameLog, *[h for h in log])
            else:
                filenameHistory = get_filename(exchangeRate, PARAMS)
                history = simulation_global(exchangeRate, PARAMS)
                np.savez_compressed('./data/' + filenameHistory, *[h for h in history])
        else:
            for e in exchanges:
                exchangeRate = round(e, 2)
                print('Exchange rate :', exchangeRate)
                filenameOut = get_filename(exchangeRate, PARAMS)
                print(filenameOut)
                if os.path.isfile(DATA_FOLDER + filenameOut):
                    with open(DATA_FOLDER + filenameOut, newline='\n') as csvfile:
                        data = csv.reader(csvfile)
                        dataInt = np.array(list(data))
                    dataInt = dataInt.astype(np.uint32)
                    extinctions = list(dataInt[:, :4])
                    final_densities = list(dataInt[:, 4])
                    last_gens = list(dataInt[:, 5])
                    last_strands = list(dataInt[:, 6])
                else:
                    extinctions, final_densities, last_gens, last_strands = [], [], [], []
                nb_already_done = len(final_densities)
                IC95 = get_IC95(extinctions)
                if any(IC95 > IC95_THRESHOLD):
                    pending = set()
                    for _ in range(n_processors):
                        pending.add(executor.submit(simulation_global, exchangeRate, make_seed()))
                    while any(IC95 > IC95_THRESHOLD) or len(extinctions) < MIN_REPETITIONS - n_processors:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            extincts, dens, lgen, last = future.result()
                            extinctions.append(extincts)
                            final_densities.append(dens)
                            last_gens.append(lgen)
                            last_strands.append(last)
                            pending.add(executor.submit(simulation_global, exchangeRate, make_seed()))
                        IC95 = get_IC95(extinctions)
                        if len(extinctions) % 50 == 0:
                            print(f'IC95 ({len(extinctions)} repetitions) :', [round(ic, 4) for ic in IC95])
                            save_results(filenameOut, extinctions, final_densities, last_gens, last_strands)
                    for future in pending:
                        extincts, dens, lgen, last = future.result()
                        extinctions.append(extincts)
                        final_densities.append(dens)
                        last_gens.append(lgen)
                        last_strands.append(last)
                    print(f'IC95 ({len(extinctions)} repetitions) :', [round(ic, 4) for ic in IC95])
                    save_results(filenameOut, extinctions, final_densities, last_gens, last_strands)