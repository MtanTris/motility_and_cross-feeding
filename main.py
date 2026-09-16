# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:37:49 2026

@author: trist
"""

import subprocess, sys, os, json
from copy import copy
import itertools as itools
from animations import show_animation
from check_params import get_filename


DEFAULT_PARAMS_PATH = r'./default_params.json'
CURRENT_PARAMS_PATH = r'./current_params.json'
SIM_PYTHON_PATH = r'./sim.py'
ANIMATIONS_PATH = r'./animations/'
DATA_FOLDER = r'./data/'


def make_history(dictCols, dictRows, generalParams, motilityRate,
                   saveLog=True, listWinners:list[list[int]] = None,
                   animate=False) -> None:
    for colParams in dictCols.values():
        for rowParams in dictRows.values():
            with open(DEFAULT_PARAMS_PATH, 'r') as f:
                defaultParams = json.load(f)
            currentParams = copy(defaultParams)
            for k,v in (list(colParams.items()) + 
                       list(rowParams.items()) + 
                       list(generalParams.items())):
                assert k in defaultParams
                currentParams[k] = v
            currentParams['SAVE_HISTORY'] = True
            if saveLog:
                currentParams['SAVE_LOG'] = True
                dataFileName = DATA_FOLDER + get_filename(motilityRate, currentParams)[0]
            else:
                dataFileName = DATA_FOLDER + get_filename(motilityRate, currentParams)
            if not os.path.isfile(dataFileName):
                print('Simulating...')
                main({'':{}}, {'':{}}, generalParams=currentParams)
            if animate:
                print('Animating...')
                show_animation(motilityRate, currentParams)
                print('Done.')
        

def main(dictRows: dict[str:dict[str:str]], dictCols: dict[str:dict[str:str]],
         generalParams:dict[str:str]={}) -> None:
    #For each combination of parameters...
    for rowParams, colParams in itools.product(dictRows.values(), dictCols.values()):
        print(rowParams)
        print(colParams)
        print(generalParams)
        #Load default parameters of the model
        with open(DEFAULT_PARAMS_PATH, 'r') as f:
            defaultParams = json.load(f)
        currentParams = copy(defaultParams)
        #Change parameters
        for k,v in (list(colParams.items()) + 
                   list(rowParams.items()) + 
                   list(generalParams.items())):
            assert k in defaultParams
            currentParams[k] = v
        #Store the changed parameters in a .json file
        with open(CURRENT_PARAMS_PATH, 'w') as f:
            json.dump(currentParams, f)
        #Run the simulations
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1' #Force unbufferized mode in sim.py (more efficient)
        process = subprocess.Popen(
            [sys.executable, '-u', f'./{SIM_PYTHON_PATH}'], #-u : unbeffered (double security)
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,                                      #No bufferization in main.py
            env=env
            )
        #Display what is printed by sim.py in the console
        for chunk in iter(lambda: process.stdout.read(256),b''):
            sys.stdout.write(chunk.decode('utf-8', errors='replace'))
            sys.stdout.flush()
        process.wait()
        

if __name__ == '__main__':
    #dictRows = {'Essential\nExtinction prob.':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0,},
    #            }
    #dictCols = {'No defectors\n' + r'$\textbf{50\times50}$ spaces':{'SPAWN_FULL_DEFECTORS':False, 'L':50},
    #            'No defectors\n' + r'$\textbf{100\times100}$ spaces (usual)':{'SPAWN_FULL_DEFECTORS':False},
    #            'No defectors\n' + r'$\textbf{200\times200}$ spaces':{'SPAWN_FULL_DEFECTORS':False, 'L':200}
    #            }
    #generalParams = {'SPAWN_CROSS_FEEDERS':False, 'GOOD':.05, 'COST':.025}
    dictRows = {#'Unessential secretions':{},
                'Unbiased\Essential':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0},
                'Chemotaxis\nEssential':{'BASAL_FITNESS':0.0, 'WEIGHT':0.0, 'CHEMOTAXIS':3},
                }
    dictCols = {'No insulators':{},
                '1x insulators':{'INSULATORS_DENSITY':1,},
                #'Twice as many':{'INSULATORS_DENSITY':2},
                #'4x insulators':{'INSULATORS_DENSITY':4},
                '10x insulators':{'INSULATORS_DENSITY':10,},
                '20x insulators':{'INSULATORS_DENSITY':20,},
                }
    generalParams = {}
    #main(dictRows, dictCols, generalParams)
    make_history(dictRows, dictCols, generalParams, 5., animate=False, saveLog=True)