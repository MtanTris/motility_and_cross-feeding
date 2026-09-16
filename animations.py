# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:12:59 2026

@author: trist
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:36:31 2026

@author: trist
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from numba import njit
from tqdm import tqdm
from copy import copy
from check_params import get_filename
import json


plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\trist\Documents\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"


colorz = ['red', 'yellow', '#00cc00', 'blue', 'black', 'gray']
labels = ['[0,0]', '[0,1]', '[1,0]', '[1,1]', 'empty', 'insulator']


@njit(cache=True)
def fitness_formula(cell_type: int, 
                    neighbours: list[int], 
                    baseRate: float, 
                    cost: list[float], 
                    good1_per_cell: list[float], 
                    good2_per_cell: list[float],
                    weight=.5,
                    RADIUS=2,
                    theta=1.) -> float:
    """
    Compute the fitness of one bacteria, using directly the equation in the 
    report.
    
    The higher the fitness, ...
    - the more resilient the bacteria is to selection events
    - the more likely the bacteria is to reproduce

    Parameters
    ----------
    cell_type : int
        Type the bacteria whose fitness we are computing.
        0: full defector
        1,2 : cross-feeder
        3: full producer
        4: empty space
    neighbours : list[int]
        Number of bacteria in the bacteria's neighboorhood, ordered by type.
    baseRate : float
        Basal fitness of all bacteria, irrespective of type and secretions.
    cost : list[float]
        Fitness lost by each cell type due to their secretion(s).
    good1_per_cell : list[float]
        Fitness gained in the vicinity of a cell secreting the first secretion.
    good2_per_cell : list[float]
        Same for the second secretion.

    Returns
    -------
    float
        Fitness of the bacteria.

    """
    #Fitness gain from secretions in the neighborhood
    arg1 = arg2 = 0.
    for i in range(5):
        arg1 = arg1 + neighbours[i]*good1_per_cell[i]
        arg2 = arg2 + neighbours[i]*good2_per_cell[i]
    #Formula : balance according to weight, add baseRate, retrieve cost
    COST = cost[2]
    fitness = baseRate - COST*(cost[cell_type]/COST)**theta + weight*max(arg1, arg2) + (1-weight)*min(arg1, arg2)
    #Fitness is capped at 0 as a lesser value would not make sense
    return max(0.,fitness)

@njit(cache=True)
def fitness(index: tuple[int],
            lattice: list[list[int]], 
            baseRate: float, 
            cost: list[float], 
            good1_per_cell: list[float],
            good2_per_cell: list[float],
            RADIUS=2,
            weight=.5,
            theta=1.) -> float:
    """
    Initiate the computation of a bacteria's fitness.
    
    The higher the fitness, ...
    - the more resilient the bacteria is to selection events
    - the more likely the bacteria is to reproduce

    Parameters
    ----------
    index : tuple[int]
        Position of the bacteria whose fitness we are computing.
    lattice : list[list[int]]
        Spatial distribution of the bacterias.
    baseRate : float
        Basal fitness of all bacteria, irrespective of type and secretions.
    cost : list[float]
        Fitness lost by each cell type due to their secretion(s).
    good1_per_cell : list[float]
        Fitness gained in the vicinity of a cell secreting the first secretion.
    good2_per_cell : list[float]
        Same for the second secretion.

    Returns
    -------
    float
        Fitness of the bacteria.

    """
    L = len(lattice[0])
    #Get the number of each type of bacteria in the neighborhood
    neighbours = [0.,0.,0.,0.,0.,0.]
    for i in range(-RADIUS,RADIUS+1):
        for j in range(-RADIUS,RADIUS+1):
            neighbours[lattice[(index[0]+i)%L, (index[1]+j)%L]] += 1.
    #Compute the fitness using the formula
    return fitness_formula(lattice[index[0], index[1]], neighbours, baseRate, 
                           cost, good1_per_cell,good2_per_cell,
                           RADIUS=RADIUS, weight=weight, theta=theta)


@njit(cache=True)
def neighbour_matrix(lattice, RADIUS=2):
    """
            [0,0] [1,0] [0,1] [1,1] empty
    [0,0]
    [1,0]
    [0,1]
    [1,1]
    """
    matrix = np.zeros((6,6))
    counts = np.array([0, 0, 0, 0, 0, 0])
    L = lattice.shape[0]
    for x in range(L):
        for y in range(L):
            cell = lattice[x, y]
            if cell != 4:
                neighbours = np.array([0,0,0,0,0,0])
                for i in range(-RADIUS,RADIUS+1):
                    for j in range(-RADIUS,RADIUS+1):
                        neighbours[lattice[(x+i)%L, (y+j)%L]] += 1
                matrix[cell] += neighbours
                counts[cell] += 1
    return matrix / counts


@njit(cache=True)
def get_avg_fitness(lattice, COST=.05, baseRate=.1, GOOD=.1, weight=.5, 
                    RADIUS=2, theta=1.):
    L = lattice.shape[0]
    good1_per_cell = [0.,GOOD,0.,GOOD,0.,0.] 
    good2_per_cell = [0.,0.,GOOD,GOOD,0.,0.]
    cost = [0.,COST,COST,2*COST,0.,0.]
    fitnesses = np.zeros(4).astype(np.float64)
    counts = np.zeros(4)
    for x in range(L):
        for y in range(L):
            cell = lattice[x, y]
            if cell != 4:
                f = fitness([x,y], lattice, baseRate, cost, good1_per_cell,
                            good2_per_cell, RADIUS=RADIUS, weight=weight,
                            theta=theta)
                fitnesses[cell] += f
                counts[cell] += 1
    return fitnesses / counts


def show_animation(exchangeRate, params, save=True, max_gen=100):
    historyFile, logFile = get_filename(exchangeRate, params)
    dataHistory = np.load('./data/' + historyFile, allow_pickle=True)
    history = [dataHistory[k] for k in dataHistory.files][:max_gen]
    dataLog = np.load('./data/' + logFile, allow_pickle=True)
    log = [dataLog[k] for k in dataLog.files][:max_gen]
    eventsHistory = event_per_gen(log)
    gens = range(len(history))
    densities = np.array([[np.sum(history[gen] == i) for i in range(6)] for gen in gens])
    fitnesses = np.array([get_avg_fitness(history[gen], COST=params['COST'], 
                                          baseRate=params['BASAL_FITNESS'], GOOD=params['GOOD'], 
                                          weight=params['WEIGHT'], RADIUS=params['RADIUS'], 
                                          theta=1.) for gen in gens])
    cmap = colors.ListedColormap(colorz)
    patches = [mpatches.Patch(color=colorz[i], label=labels[i]) for i in range(5)]
    matrixes = [neighbour_matrix(history[gen], RADIUS=params['RADIUS']) for gen in gens] 
    max_gen = max(gens) #in case it's superior
    
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(4, 10, figure=fig)
    ax_main = fig.add_subplot(gs[:, 0:4])
    ax_main.set_title("Simulation")
    ax_main.axis('off')
    
    ax_fitness = fig.add_subplot(gs[0:2, 4:6])
    ax_fitness.set_xlabel('Generation')
    ax_fitness.set_ylabel('Average fitness')
    ax_fitness.set_title('Average fitness of each strand')
    
    ax_plot = fig.add_subplot(gs[2:4, 4:6])
    ax_plot.set_title('Densities evolution')
    ax_plot.set_xlabel('Generation')
    
    ax_repr = fig.add_subplot(gs[0, 6:8])
    ax_selec = fig.add_subplot(gs[1, 6:8])
    ax_death = fig.add_subplot(gs[2, 6:8])
    ax_exch = fig.add_subplot(gs[3, 6:8])
    
    ax_00 = fig.add_subplot(gs[0, 8:10])
    ax_10 = fig.add_subplot(gs[1, 8:10])
    ax_01 = fig.add_subplot(gs[2, 8:10])
    ax_11 = fig.add_subplot(gs[3, 8:10])
    
    #Just so plt.tight_layout() takes them into account
    ax_repr.set_title('Successful reproduction events')
    ax_repr.set_xlabel('Generation')
    ax_repr.set_ylabel('Number of cells')
    
    ax_selec.set_title('Death to selection events')
    ax_selec.set_xlabel('Generation')
    ax_selec.set_ylabel('Number of cells')
    
    ax_death.set_title('Death events')
    ax_death.set_xlabel('Generation')
    ax_death.set_ylabel('Number of cells born')
    
    ax_exch.set_title('Motility of the cells')
    ax_exch.set_xlabel('Generation')
    ax_exch.set_ylabel('Number of exchange events')
    
    ax_00.set_title('Average nb of neighbours of [1,1]')
    ax_00.set_xlabel('Generation')
    ax_00.set_ylabel('Avg. nb. of neighb.')
    
    ax_10.set_title('Average nb of neighbours of [1,1]')
    ax_10.set_xlabel('Generation')
    ax_10.set_ylabel('Avg. nb. of neighb.')
    
    ax_01.set_title('Average nb of neighbours of [1,1]')
    ax_01.set_xlabel('Generation')
    ax_01.set_ylabel('Avg. nb. of neighb.')
    
    ax_11.set_title('Average nb of neighbours of [1,1]')
    ax_11.set_xlabel('Generation')
    ax_11.set_ylabel('Avg. nb. of neighb.')
    
    plt.suptitle('Animation for ' + historyFile[14:-4])
    plt.tight_layout()
    
    def update(gen: int) -> None:
        ax_main.cla()
        ax_fitness.cla()
        ax_plot.cla()
        ax_repr.cla()
        ax_selec.cla()
        ax_death.cla()
        ax_exch.cla()
        ax_00.cla()
        ax_01.cla()
        ax_10.cla()
        ax_11.cla()
    
        ax_main.imshow(history[gen], cmap=cmap, vmin=0, vmax=5, 
                       interpolation='nearest')
        ax_main.set_title("Simulation")
        ax_main.axis('off')
        
        for i in range(4):
            ax_fitness.plot(list(fitnesses[:gen,i]) + [None]*(max_gen-gen), color=colorz[i], label=labels[i])
    
        for i in range(6):
            ax_plot.plot(densities[:gen,i], color=colorz[i], label=labels[i])
        ax_plot.grid(which='both', alpha=.4)
        ax_plot.set_title('Densities evolution')
        ax_plot.set_xlabel('Generation')
        ax_plot.legend()
        ax_repr.grid(which='both', alpha=.4)
        ax_selec.grid(which='both', alpha=.4)
        ax_death.grid(which='both', alpha=.4)
        ax_exch.grid(which='both', alpha=.4)
        ax_fitness.grid(which='both', alpha=.4)
        
        ax_main.set_title("Simulation")
        ax_main.axis('off')
        ax_fitness.set_xlabel('Generation')
        ax_fitness.set_ylabel('Average fitness')
        ax_fitness.set_title('Average fitness of each strand')
        ax_plot.set_title('Densities evolution')
        ax_plot.set_xlabel('Generation')
        ax_repr.set_title('Successful reproduction events')
        ax_repr.set_xlabel('Generation')
        ax_repr.set_ylabel('Number of cells born')
        ax_selec.set_title('Death to selection events')
        ax_selec.set_xlabel('Generation')
        ax_selec.set_ylabel('Number of cells')
        ax_death.set_title('Death events')
        ax_death.set_xlabel('Generation')
        ax_death.set_ylabel('Number of cells')
        ax_exch.set_title('Motility of the cells')
        ax_exch.set_xlabel('Generation')
        ax_exch.set_ylabel('Number of exchange events')        
        
        for cellId in [0, 1, 2, 3, 5]:
            if cellId == 5 and 5 not in history[0]:
                continue
            ax_repr.plot([eventsHistory[g][cellId, 0] for g in range(gen)],
                         color=colorz[cellId])
            ax_selec.plot([eventsHistory[g][cellId, 1] for g in range(gen)],
                         color=colorz[cellId])
            ax_death.plot([eventsHistory[g][cellId, 2] for g in range(gen)],
                         color=colorz[cellId])
            ax_exch.plot([eventsHistory[g][cellId, 3] for g in range(gen)],
                         color=colorz[cellId])
            
        for cell2 in range(4):
            ax_00.plot([matrixes[g][cell2][0] for g in range(gen)], 
                     color=colorz[cell2]) 
            ax_00.grid(which='both', alpha=.4)
            ax_00.set_title('Neighbourhood of [0,0]')
            ax_00.set_xlabel('Generation')
            ax_00.set_ylabel('Avg. nb. of neighb.')
        for cell2 in range(4):
            ax_10.plot([matrixes[g][cell2][2] for g in range(gen)], 
                     color=colorz[cell2]) 
            ax_10.grid(which='both', alpha=.4)
            ax_10.set_title('Neighbourhood of [1,0]')
            ax_10.set_xlabel('Generation')
            ax_10.set_ylabel('Avg. nb. of neighb.')
        for cell2 in range(4):
            ax_01.plot([matrixes[g][cell2][1] for g in range(gen)], 
                     color=colorz[cell2]) 
            ax_01.grid(which='both', alpha=.4)
            ax_01.set_title('Neighbourhood of [0,1]')
            ax_01.set_xlabel('Generation')
            ax_01.set_ylabel('Avg. nb. of neighb.')
        for cell2 in range(4):
            ax_11.plot([matrixes[g][cell2][3] for g in range(gen)], 
                     color=colorz[cell2]) 
            ax_11.grid(which='both', alpha=.4)
            ax_11.set_title('Neighbourhood of [1,1]')
            ax_11.set_xlabel('Generation')
            ax_11.set_ylabel('Avg. nb. of neighb.')
        
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(history),
                                  interval=33)
    #if save:
    #    #FFwriter = animation.FFMpegWriter(fps=30)
    #    #ani.save('./animations/' + filename[:-4] + '.mp4', writer=FFwriter)
    #    with tqdm(total=len(history)) as pbar:
    #        FFwriter = animation.FFMpegWriter(fps=2)
    #        ani.save('./animations/' + historyFile[:-4] + '.avi', fps=2,
    #                 progress_callback=lambda i,n:pbar.update(1),)
    if save:
        # Fixe une taille précise et paire
        fig.set_size_inches(20, 8)
        dpi = 100  # 20*100=2000 (pair), 8*100=800 (pair)
        
        FFwriter = animation.FFMpegWriter(fps=4, codec='libx264',
                                            extra_args=['-pix_fmt', 'yuv420p'])
        with tqdm(total=len(history)) as pbar:
            ani.save('./animations/' + historyFile[:-4] + '.mp4', writer=FFwriter,
                      dpi=dpi,
                      savefig_kwargs={'facecolor': 'white', 'bbox_inches': None},
                      progress_callback=lambda i, n: pbar.update(1))
    return ani


@njit(cache=True)
def event_per_gen(log: list[list[tuple]]) -> list:
    out = []
    for gen in range(len(log)):
        out_gen = np.zeros((5, 4)) #reproduction, selection, death, exchange
        for event in log[gen]:
            if event[0] == 0: #reproduction
                out_gen[event[1], 0] += 1
            elif event[0] == 1: #selection
                out_gen[event[1], 1] += 1
            elif event[0] == 2: #death
                out_gen[event[1], 2] += 1
            elif event[0] == 3: #exchange
                out_gen[event[1], 3] += 1
                out_gen[event[4], 3] += 1
        out.append(out_gen)
    return out

DEFAULT_PARAMS_PATH = r'./default_params.json'
    