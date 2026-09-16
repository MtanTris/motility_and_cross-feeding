# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:27:29 2026

@author: trist
"""

import json

DEFAULT_PARAMS_PATH = r'./default_params.json'

def get_filename(exchangeRate, PARAMS):
    with open(DEFAULT_PARAMS_PATH, 'r') as f:
        defaultParams = json.load(f)
    for k in PARAMS.keys():
        assert k in defaultParams
    out = f'ER {exchangeRate}'
    if PARAMS['L'] != 100:
        out += f' L={PARAMS["L"]}'
    if PARAMS['SELECTION_RATE'] != 1:
        out += f' sR={PARAMS["SELECTION_RATE"]}'
    if PARAMS['REPRODUCTION_RATE'] != 1:
        out += f' rR={PARAMS["REPRODUCTION_RATE"]}'
    if PARAMS['DEATH_RATE'] != .05:
        out += f' dR={PARAMS["DEATH_RATE"]}'
    if PARAMS['BASAL_FITNESS'] != .1:
        out += f' BF={PARAMS["BASAL_FITNESS"]}'
    if PARAMS['GOOD'] != .1:
        out += f' G={PARAMS["GOOD"]}'
    if PARAMS['COST'] != .05:
        out += f' C={PARAMS["COST"]}'
    if PARAMS['WEIGHT'] != .5:
        out += f' W={PARAMS["WEIGHT"]}'
    if PARAMS['INITIAL_SPARSITY'] != 50:
        out += f' IS={PARAMS["INITIAL_SPARSITY"]}'
    if PARAMS['RADIUS'] != 2:
        out += f' R={PARAMS["RADIUS"]}'
    if PARAMS['DETECTION_RADIUS'] != 3:
        out += f' DR={PARAMS["DETECTION_RADIUS"]}'
    if PARAMS['CHEMOTAXIS'] != 0:
        out += f' CHTX={PARAMS["CHEMOTAXIS"]}'
    if PARAMS['BETA'] != .2:
        out += f' B={PARAMS["BETA"]}'
    if PARAMS['BLOCKING']:
        out += f' BLK={PARAMS["BLOCKING"]}'
    if PARAMS['MIN_EXCHANGE_RATE'] is not None:
        out += f' MIN_ER={PARAMS["MIN_EXCHANGE_RATE"]}'
    if PARAMS['DD_THRESHOLD'] != .5:
        out += f' DD_T={PARAMS["DD_THRESHOLD"]}'
    if not PARAMS['SPAWN_FULL_PRODUCERS']:
        out += f' SPW_FP={PARAMS["SPAWN_FULL_PRODUCERS"]}'
    if not PARAMS['SPAWN_CROSS_FEEDERS']:
        out += f' SPW_CF={PARAMS["SPAWN_CROSS_FEEDERS"]}'
    if not PARAMS['SPAWN_FULL_DEFECTORS']:
        out += f' SPW_FD={PARAMS["SPAWN_FULL_DEFECTORS"]}'
    if PARAMS['INSULATORS_DENSITY'] != 0:
        out += f' INS_D={PARAMS["INSULATORS_DENSITY"]}'
    if PARAMS['SAVE_LOG']:
        return f'history {out}.npz', f'log {out}.npz'
    elif PARAMS['SAVE_HISTORY']:
        return f'history {out}.npz'
    else:
        return f'out {out}.csv'