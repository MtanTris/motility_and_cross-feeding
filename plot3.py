# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:40:50 2026

@author: trist
"""

import numpy as np
import matplotlib.pyplot as plt
from check_params import get_filename
import csv
import os
import json
from copy import copy
from matplotlib import colors as pltcolors
from numba import njit

colors = ['red', 'yellow', '#00cc00', 'blue', 'black', 'gray']
labels_2sec = ['[0,0]', '[0,1]', '[1,0]', '[1,1]', 'empty', 'insulator']
labels_1sec = ['[0]', None, None, '[1]', 'empty', 'insulator']
labels_strains = ['defectors', None, 'cross-feeders', 'producers']

DEFAULT_PARAMS_PATH = r'./default_params.json'
DATA_FOLDER = r'./data/'
FIGURES_FOLDER = r'../Report figures/'

epsilonMax = 20.
epsilonStep = .5
EPSILONS = np.arange(0.0, epsilonMax+epsilonStep, epsilonStep)
ZOOM = .48
with open(DEFAULT_PARAMS_PATH, 'r') as f:
    defaultParams = json.load(f)
    

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


def make_errorbar_line(data: dict[float:list[float]], color: str, label: str,
                       maxY=1., minY=0.) -> None:
    X = sorted(data)
    means = [np.mean(data[x]) for x in X]
    IC95 = [1.96*np.std(data[x])/len(data[x])**.5 for x in X]
    uppers = [IC95[i] if means[i]+IC95[i] <= maxY else maxY - means[i]
              for i in range(len(X))]
    lowers = [IC95[i] if means[i]-IC95[i] >= minY else minY + means[i]
              for i in range(len(X))]
    plt.errorbar(X, means, yerr=[lowers, uppers], fmt='o-', capsize=5,
                 color=color, label=label)
    plt.ylim([minY-.03, maxY+.03])
    plt.grid(which='both', alpha=.4)


def make_boxplot(data: dict[float:[list[float]]]) -> None:
    epsilons = sorted(data)
    values = list(data.values())
    plt.boxplot(values, positions=epsilons, manage_ticks=False, whis=0,
                sym='.', showcaps=False, showbox=False)
    plt.grid(which='both', alpha=.4)
    


def make_subplot(params: dict[str:], dtype: int, motilityRate: float = None,
                 gen: int = None) -> None:
    dataWhole = {}
    for epsilon in EPSILONS:
        e = round(epsilon, 2)
        filename = get_filename(e, params)
        if os.path.isfile(DATA_FOLDER + filename):
            with open(DATA_FOLDER + filename, newline='\n') as f:
                data = csv.reader(f)
                dataInt = np.array(list(data))
            dataInt = dataInt.astype(np.uint32)
            dataWhole[e] = dataInt
    epsilons = sorted(dataWhole)  # get rid of keys that don't exist
    if dtype == 0:  # extinction probabilities
        if not params['SPAWN_CROSS_FEEDERS'] and params['GOOD'] == .05:
            labels = labels_1sec
        else:
            labels = labels_2sec
        for i in range(4):
            if not params['SPAWN_FULL_DEFECTORS'] and i == 0:
                continue
            elif not params['SPAWN_CROSS_FEEDERS'] and i in (1, 2):
                continue
            elif not params['SPAWN_FULL_PRODUCERS'] and i == 3:
                continue
            data_i = {e: [dataWhole[e][k][i] for k in range(len(dataWhole[e]))]
                      for e in epsilons}
            make_errorbar_line(data_i, colors[i], labels[i])
        plt.ylabel('Extinction probability')
        plt.xlabel(r'Exchange rate $\varepsilon$')
        plt.xlim([0.0, epsilonMax])
        plt.legend()
    elif dtype == 1:  # nb of cells at time of last extinction
        data = {e: [dataWhole[e][k][4] for k in range(len(dataWhole[e]))]
                for e in epsilons}
        make_boxplot(data)
        plt.ylabel('Final population size')
        plt.ylim([0, params['L']**2])
        plt.xlabel(r'Exchange rate $\varepsilon$')
        plt.xlim([0.0, epsilonMax])
    elif dtype == 2:  # generation number for time of last extinction
        data = {e: [dataWhole[e][k][5] for k in range(len(dataWhole[e]))]
                for e in epsilons}
        make_boxplot(data)
        plt.ylabel('Final generation')
        plt.xlabel(r'Exchange rate $\varepsilon$')
        plt.xlim([0.0, epsilonMax])
    elif dtype == 3:  # probability of total extinction
        extinctions = {e: [dataWhole[e][k][:4] for k in range(len(dataWhole))]
                       for e in epsilons}
        data = {e: [0 in extinctions[e][k]
                    for k in range(len(extinctions[e]))]
                for e in epsilons}
        make_errorbar_line(data, 'black', '')
        plt.ylabel('Probability of total extinction')
        plt.xlabel(r'Exchange rate $\varepsilon$')
        plt.xlim([0.0, epsilonMax])
    elif dtype == 4:  # probability to be the last one standing
        labels = labels_strains
        lasts = {e: [dataWhole[e][k][6] for k in range(len(dataWhole[e]))]
                 for e in epsilons}
        for i in (0, 1, 3):
            if not params['SPAWN_FULL_DEFECTORS'] and i == 0:
                continue
            elif not params['SPAWN_CROSS_FEEDERS'] and i == 1:
                continue
            elif not params['SPAWN_FULL_PRODUCERS'] and i == 3:
                continue
            is_i_last = {e: [lasts[e][k] != i for k in range(len(lasts[e]))]
                         for e in epsilons}
            make_errorbar_line(
                is_i_last, colors[i] if colors[i] != 'yellow' else '#00cc00', labels[i])
        plt.legend()
        plt.ylabel('Prob. to not be the last strand')
        plt.xlabel(r'Exchange rate $\varepsilon$')
        plt.xlim([0.0, epsilonMax])
    elif dtype == 5:  # number of replicas
        data = [len(dataWhole[e]) for e in epsilons]
        plt.plot(epsilons, data, 'k+')
        plt.ylabel('Number of replicates')
        plt.grid(which='both', alpha=.4)
    elif dtype in (6, 7):
        # Load simulation history
        params['SAVE_HISTORY'] = True
        params['SAVE_LOG'] = False
        assert motilityRate is not None
        historyFile = get_filename(motilityRate, params)
        print(historyFile)
        if os.path.isfile(DATA_FOLDER + historyFile):
            dataHistory = np.load(DATA_FOLDER + historyFile, allow_pickle=True)
            history = [dataHistory[k] for k in dataHistory.files]
        if dtype == 6:  # densities history
            gens = range(len(history))
            # Compute the densities
            densities = np.array([[np.sum(history[gen] == i)
                                 for gen in gens] for i in range(6)])
            plot_densities_history(params, densities)
            plt.xlabel('Generation')
            plt.grid(which='both', alpha=.4)
            plt.legend()
        elif dtype == 7:
            assert gen is not None
            cmap = pltcolors.ListedColormap(colors)
            plt.imshow(history[gen], cmap=cmap, vmin=0, vmax=5,
                       interpolation='none')
            #plt.axis('off')
            plt.xticks([])
            plt.yticks([])
    elif dtype in (8,9):
        params['SAVE_LOG'], params['SAVE_HISTORY'] = True, True
        assert motilityRate is not None
        _, logFile = get_filename(motilityRate, params)
        print(logFile)
        dataLog = np.load(DATA_FOLDER + logFile, allow_pickle=True)
        log = [dataLog[k] for k in dataLog.files]
        eventsHistory = event_per_gen(log)
        max_gen = len(eventsHistory)
        if not params['SPAWN_CROSS_FEEDERS'] and params['GOOD'] == .05:
            labels = labels_1sec
        else:
            labels = labels_2sec
        for cellId in [0, 1, 2, 3, 5]:
            check = True
            if cellId == 5:
                check = False
            if cellId in (1,2) and not params['SPAWN_CROSS_FEEDERS'] and params['GOOD'] == .05:
                check = False
            if check:
                if dtype == 8:
                    plt.plot([eventsHistory[g][cellId, 1] for g in range(max_gen)],
                                 color=colors[cellId], label=labels[cellId])
                    plt.xlabel('Generation')
                    plt.ylabel('Nb. of selection\nevents lost')
                elif dtype == 9:
                    plt.plot([eventsHistory[g][cellId, 0] for g in range(max_gen)],
                                 color=colors[cellId], label=labels[cellId])
                    plt.xlabel('Generation')
                    plt.ylabel('Nb. of successful\nreproduction events')
        plt.grid(which='both', alpha=.4)
        plt.legend()


"""
dtype : int, optional
    Type of data to be plotted. The default is 0.
    0 -> extinction probabilities
    1 -> nb of cells at the time when the second strand goes extinct
    2 -> generation of the extinction of the second strand
    3 -> probability that all strands go extinct
    4 -> probability that one strand is the last one standing
    5 -> number of replicas for each motility rate
"""


def dispatch_plots(dictRows: dict[str:], dictCols: dict[str:],
                   generalParams={}, suptitle='', grayedRows=[],
                   grayedCols=[], pad=5, zoom=ZOOM):
    ncols, nrows = len(dictCols), len(dictRows)
    width_ratios = [4.8 if d.get('dtype',0)==7 else 6.4 for d in dictCols.values()]
    height_ratios = [4.8 for _ in dictRows.values()]
    print(width_ratios, height_ratios)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,width_ratios=width_ratios,
                             height_ratios=height_ratios,
                             figsize=(sum(width_ratios)*zoom, sum(height_ratios)*zoom))
    graphCount = 1
    for rowNumber, (rowName, rowParams) in enumerate(dictRows.items()):
        for colNumber, (colName, colParams) in enumerate(dictCols.items()):
            params = copy(defaultParams)
            newParams = colParams | rowParams | generalParams
            dtype = newParams.pop('dtype')
            motilityRate = newParams.pop('motilityRate', None)
            gen = newParams.pop('gen', None)
            for paramName, paramValue in newParams.items():
                assert paramName in defaultParams
                params[paramName] = paramValue
            ax = plt.subplot(nrows, ncols, graphCount)
            make_subplot(params, dtype, motilityRate=motilityRate, gen=gen)
            graphCount += 1
    if ncols > 1:
        if nrows > 1:
            for ax, colName in zip(axes[0], dictCols.keys()):
                ax.annotate(colName, xy=(0.5, 1), xytext=(0, pad),
                            xycoords='axes fraction', textcoords='offset points',
                            size='larger', ha='center', va='baseline', fontweight='bold')
            for ax, rowName in zip(axes[:, 0], dictRows.keys()):
                ax.annotate(rowName, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='larger', ha='right', va='center', fontweight='bold',
                            rotation=90)
        else:
            for ax, colName in zip(axes, dictCols.keys()):
                ax.annotate(colName, xy=(0.5, 1), xytext=(0, pad),
                            xycoords='axes fraction', textcoords='offset points',
                            size='larger', ha='center', va='baseline', fontweight='bold')
            for ax, rowName in zip(axes, dictRows.keys()):
                ax.annotate(rowName, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='larger', ha='right', va='center', fontweight='bold',
                            rotation=90)
    else:
        if nrows > 1:
            for ax, colName in zip(axes, dictCols.keys()):
                ax.annotate(colName, xy=(0.5, 1), xytext=(0, pad),
                            xycoords='axes fraction', textcoords='offset points',
                            size='larger', ha='center', va='baseline', fontweight='bold')
            for ax, rowName in zip(axes, dictRows.keys()):
                ax.annotate(rowName, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='larger', ha='right', va='center', fontweight='bold',
                            rotation=90)
        else:
            for colName in dictCols.keys():
                axes.annotate(colName, xy=(0.5, 1), xytext=(0, pad),
                            xycoords='axes fraction', textcoords='offset points',
                            size='larger', ha='center', va='baseline', fontweight='bold')            
    fig.tight_layout()
    #plt.subplots_adjust(wspace=1, hspace=0)
    #plt.subplots_adjust(pad=0)


def plot_densities_history(params, densities):
    genotypes = [0, 1, 2, 3]
    if params['INSULATORS_DENSITY'] > 0:
        genotypes.append(5)
    if not params['SPAWN_CROSS_FEEDERS'] and params['GOOD'] == .05:
        labels = labels_1sec
    else:
        labels = labels_2sec
    # Plotting
    for i in genotypes:
        if not params['SPAWN_FULL_DEFECTORS'] and i == 0:
            continue
        elif not params['SPAWN_CROSS_FEEDERS'] and i in (1, 2):
            continue
        elif not params['SPAWN_FULL_PRODUCERS'] and i == 3:
            continue
        plt.plot(densities[i], color=colors[i], label=labels[i])
    plt.xlabel('Generation')
    plt.ylabel('Number of cells')
    plt.grid(which='both', alpha=.4)
    plt.legend()
    

def figure2a():
    dictRows = {'Unessential\nsecretions': {},
                'Essential\nsecretions': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    dictCols = {'Extinction\nprobabilities': {'dtype': 0},
                #'Spatial structures\n' + r'$\mathbf{\varepsilon=2}$':{'motilityRate':2., 'gen':30, 'dtype':7},
                #'Spatial structures\n' + r'$\mathbf{\varepsilon=15}$':{'motilityRate':15., 'gen':25, 'dtype':7},
                #'Final number\nof cells': {'dtype': 1},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS': False, 'GOOD': .05, 'COST': .025}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_2a.pdf')
    
    
def figure2b():
    dictRows = {'': {},
                ' ': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    dictCols = {#'Extinction\nprobabilities': {'dtype': 0},
                #'Spatial structures\n' + r'$\mathbf{\varepsilon=2}$':{'motilityRate':2., 'gen':30, 'dtype':7},
                #'Spatial structures\n' + r'$\mathbf{\varepsilon=15}$':{'motilityRate':15., 'gen':25, 'dtype':7},
                'Final number\nof cells': {'dtype': 1},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS': False, 'GOOD': .05, 'COST': .025}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_2b.pdf')
    
    
def figure2c():
    dictRows = {'Unessential\nsecretions': {},
                'Essential\nsecretions': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    dictCols = {#'Extinction\nprobabilities': {'dtype': 0},
                'Spatial structures\n' + r'$\mathbf{\varepsilon=2}$':{'motilityRate':2., 'gen':30, 'dtype':7},
                'Spatial structures\n' + r'$\mathbf{\varepsilon=15}$':{'motilityRate':15., 'gen':25, 'dtype':7},
                #'Final number\nof cells': {'dtype': 1},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS': False, 'GOOD': .05, 'COST': .025}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_2c.pdf')
    
    
def figure2d():
    dictRows = {'Essential\nExtinction prob.':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0,},
                }
    dictCols = {'No defectors\n' + r'$\mathbf{50\times50}$ spaces':{'SPAWN_FULL_DEFECTORS':False, 'L':50},
                'No defectors\n' + r'$\mathbf{100\times100}$ spaces (usual)':{'SPAWN_FULL_DEFECTORS':False},
                'No defectors\n' + r'$\mathbf{200\times200}$ spaces':{'SPAWN_FULL_DEFECTORS':False, 'L':200}
                }
    generalParams = {'SPAWN_CROSS_FEEDERS':False, 'GOOD':.05, 'COST':.025, 'dtype':0}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_2d.pdf')
    
    
def figure2e():
    dictRows = {'Unessential\nsecretions': {},
                'Essential\nsecretions': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    dictCols = {'Reproduction events\n' + r'$\mathbf{\varepsilon=2}$':{'motilityRate':2., 'dtype':9},
                'Reproduction events\n' + r'$\mathbf{\varepsilon=15}$':{'motilityRate':15., 'dtype':9},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS': False, 'GOOD': .05, 'COST': .025}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_2e.pdf')
    
    
def figure2f():
    dictRows = {'': {},
                ' ': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    dictCols = {'Selection events\n' + r'$\mathbf{\varepsilon=2}$':{'motilityRate':2., 'dtype':8},
                'Selection events\n' + r'$\mathbf{\varepsilon=15}$':{'motilityRate':15., 'dtype':8},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS': False, 'GOOD': .05, 'COST': .025}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_2f.pdf')
    
    
def figure3a():
    dictCols = {'Extinction prob.\nUnbiased motility':{'dtype':0},
                'Extinction prob.\nChemotaxis':{'dtype':0, 'CHEMOTAXIS':3},
                #'Spatial structures\nChemotaxis\n' + r'$\mathbf{\varepsilon=1}$':{'dtype':7, 'motilityRate':1., 'CHEMOTAXIS':3, 'gen':15}, #gen:50
                #'Spatial structures\nChemotaxis\n' + r'$\mathbf{\varepsilon=15}$':{'dtype':7, 'motilityRate':15., 'CHEMOTAXIS':3, 'gen':15},
               # 'Final pop. size\nChemotaxis':{'dtype':1, 'CHEMOTAXIS':3},
                }
    dictRows = {'Unessential\nsecretions': {},
                'Essential\nsecretions': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS':False, 'GOOD':.05, 'COST':.025}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_3a.pdf')
    
    
def figure3b():
    dictCols = {'Final pop. size\nUnbiased motility':{'dtype':1},
                'Final pop. size\nChemotaxis':{'dtype':1, 'CHEMOTAXIS':3},
                #'Spatial structures\nChemotaxis\n' + r'$\mathbf{\varepsilon=1}$':{'dtype':7, 'motilityRate':1., 'CHEMOTAXIS':3, 'gen':15}, #gen:50
                #'Spatial structures\nChemotaxis\n' + r'$\mathbf{\varepsilon=15}$':{'dtype':7, 'motilityRate':15., 'CHEMOTAXIS':3, 'gen':15},
               # 'Final pop. size\nChemotaxis':{'dtype':1, 'CHEMOTAXIS':3},
                }
    dictRows = {'': {},
                ' ': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS':False, 'GOOD':.05, 'COST':.025}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_3b.pdf')
    
    
def figure3c():
    dictCols = {#'Final pop. size\nUnbiased motility':{'dtype':1},
                #'Final pop. size\nChemotaxis':{'dtype':1, 'CHEMOTAXIS':3},
                'Spatial structures\nChemotaxis\n' + r'$\mathbf{\varepsilon=1}$':{'dtype':7, 'motilityRate':1., 'CHEMOTAXIS':3, 'gen':50}, #gen:50
                'Spatial structures\nChemotaxis\n' + r'$\mathbf{\varepsilon=15}$':{'dtype':7, 'motilityRate':15., 'CHEMOTAXIS':3, 'gen':15},
               # 'Final pop. size\nChemotaxis':{'dtype':1, 'CHEMOTAXIS':3},
                }
    dictRows = {'Unessential\nsecretions': {},
                'Essential\nsecretions': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS':False, 'GOOD':.05, 'COST':.025}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[])
    plt.savefig(FIGURES_FOLDER + 'figure_3c.pdf')
    
    
def figure3c2():
    dictRows = {'Unessential\nsecretions': {},
                'Essential\nsecretions': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    dictCols = {#'Extinction\nprobabilities': {'dtype': 0},
                'Unbiased motility\n' + r'$\mathbf{\varepsilon=2}$':{'motilityRate':2., 'gen':40, 'dtype':7},
                'Unbiased motility\n' + r'$\mathbf{\varepsilon=10}$':{'motilityRate':10., 'gen':25, 'dtype':7},
                'Chemotaxis\n' + r'$\mathbf{\varepsilon=2}$':{'motilityRate':2., 'gen':40, 'dtype':7, 'CHEMOTAXIS':3},
                'Chemotaxis\n' + r'$\mathbf{\varepsilon=10}$':{'motilityRate':10., 'gen':25, 'dtype':7, 'CHEMOTAXIS':3},
                #'Final number\nof cells': {'dtype': 1},
                }
    generalParams = {'SPAWN_CROSS_FEEDERS': False, 'GOOD': .05, 'COST': .025}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_3c2.pdf')
    
    
def figure3d():
    dictRows = {r'$\mathbf{\varepsilon=2}$':{'motilityRate':2.},
                r'$\mathbf{\varepsilon=15}$':{'motilityRate':15.}
                }
    dictCols = {'Selection events':{'CHEMOTAXIS':3, 'dtype':8, 'BASAL_FITNESS':0.0, 'WEIGHT':0.0}
                }
    generalParams = {'SPAWN_CROSS_FEEDERS': False, 'GOOD': .05, 'COST': .025}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_3d.pdf')
    
    
def figure4a():
    dictCols = {r'$\mathbf{50\times50}$ spaces':{'L':50},
                r'$\mathbf{100\times100}$ spaces (usual)':{},
                r'$\mathbf{200\times200}$ spaces':{'L':200},
                r'$\mathbf{300\times300}$ spaces':{'L':300},
                }
    dictRows = {'Unessential\nsecretions':{},
                'Essential\nsecretions':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0},
                }
    generalParams = {'dtype':0}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[])
    plt.savefig(FIGURES_FOLDER + 'figure_4a.pdf')
    
    
def figure4b():
    dictRows = {#'Unessential\nsecretions': {},
                'Essential\n' + r'$\mathbf{100\times100}$ spaces': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    dictCols = {#'Extinction\nprobabilities':{'dtype':0},
                r'$\mathbf{\varepsilon=2}$':{'motilityRate':2., 'dtype':7},
                r'$\mathbf{\varepsilon=5}$':{'motilityRate':5., 'dtype':7},
                r'$\mathbf{\varepsilon=10}$':{'motilityRate':10., 'dtype':7},
                r'$\mathbf{\varepsilon=15}$':{'motilityRate':15., 'dtype':7},
                #'Final number\nof cells': {'dtype': 1},
                }
    generalParams = {'gen':45}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_4b.pdf')
    

def figure4c():
    dictRows = {'Essential\n' + r'$\mathbf{100\times100}$ spaces': {'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                }
    dictCols = {'No cross-feeders':{'SPAWN_CROSS_FEEDERS':False},
                'No defectors':{'SPAWN_FULL_DEFECTORS':False},
                'No producers':{'SPAWN_FULL_PRODUCERS':False}
                }
    generalParams = {'dtype':4}
    dispatch_plots(dictRows, dictCols, generalParams)
    plt.savefig(FIGURES_FOLDER + 'figure_4c.pdf')


def figure4d():
    dictRows = {'':{}}
    dictCols = {'Cross-feeders only':{'SPAWN_FULL_PRODUCERS':False},
                'Producers only':{'SPAWN_CROSS_FEEDERS':False},
                }
    generalParams = {'dtype':0, 'SPAWN_FULL_DEFECTORS':False, 'BASAL_FITNESS':0.0, 'WEIGHT':0.0}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[])
    plt.savefig(FIGURES_FOLDER + 'figure_4d.pdf')
    
    
def figure4e():
    dictRows = {#r'$\mathbf{50\times50}$ spaces':{'L':50},
                '':{},
                #r'$\mathbf{200\times200}$ spaces':{'L':200},
                #r'$\mathbf{300\times300}$ spaces':{'L':300},
                }
    dictCols = {'Unessential\nsecretions':{},
                'Essential\nsecretions':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0},
                }
    generalParams = {'dtype':0}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[])
    plt.savefig(FIGURES_FOLDER + 'figure_4e.pdf')
    
    
def figure5a():
    dictRows = {'Unessential\nExtinction prob.': {'dtype':0},
                'Essential\nExtinction prob.': {'dtype':0, 'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0},
                #'Essential\nTime dynamics\n' + r'$\mathbf{\varepsilon=15}$':{'dtype':6, 'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0, 'motilityRate':15.},
                #'Essential\nSpatial structures\n' + r'$\mathbf{\varepsilon=15}$':{'dtype':7, 'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0, 'motilityRate':15., 'gen':50}
                }
    dictCols = {'Unbiased\nmotility':{},
                'Chemotaxis towards\nwhat the genotype\ndoes not produces':{'CHEMOTAXIS':1},
                'Chemotaxis towards\nwhat the genotype\ndoes produce':{'CHEMOTAXIS':2},
                'Chemotaxis towards\nall secretions':{'CHEMOTAXIS':3},
                }
    generalParams = {} 
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_5a.pdf')
    
    
def figure5b():
    dictRows = {'Essential\nTime dynamics\n' + r'$\mathbf{\varepsilon=15}$':{'dtype':6, 'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0, 'motilityRate':15.},
                #'Essential\nSpatial structures\n' + r'$\mathbf{\varepsilon=15}$':{'dtype':7, 'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0, 'motilityRate':15., 'gen':50}
                }
    dictCols = {'Unbiased\nmotility':{},
                'Chemotaxis towards\nwhat the genotype\ndoes not produces':{'CHEMOTAXIS':1},
                'Chemotaxis towards\nwhat the genotype\ndoes produce':{'CHEMOTAXIS':2},
                'Chemotaxis towards\nall secretions':{'CHEMOTAXIS':3},
                }
    generalParams = {} 
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_5b.pdf')
    
    
def figure5c():
    dictRows = {#'Essential\nTime dynamics\n' + r'$\mathbf{\varepsilon=15}$':{'dtype':6, 'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0, 'motilityRate':15.},
                'Essential\nSpatial structures\n' + r'$\mathbf{\varepsilon=15}$':{'BASAL_FITNESS': 0.0, 'WEIGHT': 0.0, 'motilityRate':15., 'gen':50}
                }
    dictCols = {'Unbiased\nmotility':{'dtype':7},
                'Chemotaxis towards\nwhat the genotype\ndoes not produces':{'CHEMOTAXIS':1, 'dtype':7},
                'Chemotaxis towards\nwhat the genotype\ndoes produce':{'CHEMOTAXIS':2, 'dtype':7},
                'Chemotaxis towards\nall secretions':{'CHEMOTAXIS':3, 'dtype':7},
                }
    generalParams = {} 
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_5c.pdf')
    
    
def figure5d():
    dictRows = {'Essential':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0}}
    dictCols = {'CHEMOTAXIS2':{'CHEMOTAXIS':2}}
    generalParams = {'dtype':0, 'SPAWN_FULL_PRODUCERS':False, 'SPAWN_FULL_DEFECTORS':False}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_5d.pdf')
    
    
def figure6a():
    dictRows = {'Unessential\nExtinction prob.':{'dtype':0},
                #'Unessential\nSpatial structures\n' + r'$\mathbf{\varepsilon=3}$':{'dtype':7, 'motilityRate':3., 'gen':50},
                #'Unessential\nSpatial structures\n' + r'$\mathbf{\varepsilon=3}$' +' other gen':{'dtype':6, 'motilityRate':3., 'gen':20},
                'Essential\nExtinction prob.':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'dtype':0},
                #'Essential\nSpatial structures\n' + r'$\mathbf{\varepsilon=6}$':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'dtype':7, 'motilityRate':6., 'gen':80}
                }
    dictCols = {#'Threshold=.2':{'DD_THRESHOLD':.2},
                #'Threshold=.4':{'DD_THRESHOLD':.4},
                'Threshold=0.5':{'DD_THRESHOLD':.5},
                #'Threshold=.6':{'DD_THRESHOLD':.6},
                'Threshold=0.7':{'DD_THRESHOLD':.7},
                'Threshold=0.8':{'DD_THRESHOLD':.8},
                'Threshold=.9':{'DD_THRESHOLD':.9},
                'Threshold=1.0':{'DD_THRESHOLD':1.},
                }
    generalParams = {'MIN_EXCHANGE_RATE':.1}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[-1])
    plt.savefig(FIGURES_FOLDER + 'figure_6a.pdf')
    
    
def figure6b():
    dictRows = {#'Unessential\nExtinction prob.':{'dtype':0},
                'Unessential\nSpatial structures\n' + r'$\mathbf{\varepsilon=3}$':{'dtype':7, 'motilityRate':3., 'gen':50},
                #'Unessential\nSpatial structures\n' + r'$\mathbf{\varepsilon=3}$' +' other gen':{'dtype':6, 'motilityRate':3., 'gen':20},
                #'Essential\nExtinction prob.':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'dtype':0},
                'Essential\nSpatial structures\n' + r'$\mathbf{\varepsilon=6}$':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'dtype':6, 'motilityRate':6., 'gen':40}
                }
    dictCols = {#'Threshold=.2':{'DD_THRESHOLD':.2},
                #'Threshold=.4':{'DD_THRESHOLD':.4},
                'Threshold=0.5':{'DD_THRESHOLD':.5},
                #'Threshold=.6':{'DD_THRESHOLD':.6},
                'Threshold=0.7':{'DD_THRESHOLD':.7},
                'Threshold=0.8':{'DD_THRESHOLD':.8},
                #'Threshold=.9':{'DD_THRESHOLD':.9},
                'Threshold=1.0':{'DD_THRESHOLD':1.},
                }
    generalParams = {'MIN_EXCHANGE_RATE':.1}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[-1])
    plt.savefig(FIGURES_FOLDER + 'figure_6b.pdf')
    
    
def figure6c():
    dictRows = {#'Unessential\nExtinction prob.':{'dtype':0},
                #'Unessential\nSpatial structures\n' + r'$\mathbf{\varepsilon=3}$':{'dtype':7, 'motilityRate':3., 'gen':50},
                #'Unessential\nSpatial structures\n' + r'$\mathbf{\varepsilon=3}$' +' other gen':{'dtype':6, 'motilityRate':3., 'gen':20},
                #'Essential\nExtinction prob.':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'dtype':0},
                'Essential\nTime dynamics\n' + r'$\mathbf{\varepsilon=6}$':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'dtype':6, 'motilityRate':6., 'gen':40}
                }
    dictCols = {#'Threshold=.2':{'DD_THRESHOLD':.2},
                #'Threshold=.4':{'DD_THRESHOLD':.4},
                'Threshold=0.5':{'DD_THRESHOLD':.5},
                #'Threshold=.6':{'DD_THRESHOLD':.6},
                'Threshold=0.7':{'DD_THRESHOLD':.7},
                'Threshold=0.8':{'DD_THRESHOLD':.8},
                #'Threshold=.9':{'DD_THRESHOLD':.9},
                'Threshold=1.0':{'DD_THRESHOLD':1.},
                }
    generalParams = {'MIN_EXCHANGE_RATE':.1}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[-1])
    plt.savefig(FIGURES_FOLDER + 'figure_6c.pdf')
    
    
def figure7a():
    dictRows = {#'Unessential secretions':{},
                'Unbiased\nEssential':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0},
                'Chemotaxis\nEssential':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'CHEMOTAXIS':3},
                }
    dictCols = {'No insulators':{},
                '1x insulators':{'INSULATORS_DENSITY':1},
                #'Twice as many':{'INSULATORS_DENSITY':2},
                #'4x insulators':{'INSULATORS_DENSITY':4},
                '10x insulators':{'INSULATORS_DENSITY':10},
                '20x insulators':{'INSULATORS_DENSITY':20},
                }
    generalParams = {'dtype':0}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_7a.pdf')



def figure7a():
    dictRows = {#'Unessential secretions':{},
                'Unbiased\nEssential':{},
                'Chemotaxis\nEssential':{'CHEMOTAXIS':3},
                }
    dictCols = {'No insulators':{'dtype':0},
                '1x insulators':{'INSULATORS_DENSITY':1,'dtype':0},
                #'Twice as many':{'INSULATORS_DENSITY':2},
                #'4x insulators':{'INSULATORS_DENSITY':4},
                '10x insulators':{'INSULATORS_DENSITY':10,'dtype':0},
                '20x insulators':{'INSULATORS_DENSITY':20,'dtype':0},
                }
    generalParams = {}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_7a.pdf')
    
    
def figure7b():
    dictRows = {#'Unessential secretions':{},
                'Unbiased\nEssential':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0},
                'Chemotaxis\nEssential':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'CHEMOTAXIS':3},
                }
    dictCols = {'No insulators':{'dtype':7},
                '1x insulators':{'INSULATORS_DENSITY':1,'dtype':7},
                #'Twice as many':{'INSULATORS_DENSITY':2},
                #'4x insulators':{'INSULATORS_DENSITY':4},
                '10x insulators':{'INSULATORS_DENSITY':10,'dtype':7},
                '20x insulators':{'INSULATORS_DENSITY':20,'dtype':7},
                }
    generalParams = {'motilityRate':5., 'gen':20}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_7b.pdf')


def figure_poster_unbiased():
    dictRows = {'':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0}}
    dictCols = {'':{'dtype':0}}
    generalParams = {}
    dispatch_plots(dictRows, dictCols, generalParams, grayedCols=[0])
    plt.savefig(FIGURES_FOLDER + 'figure_poster_unbiased.pdf')
    
    
if __name__ == '__main__':
    figure_poster_unbiased()