# Introduction to the model
This repository stores the code for a cellular automata model aiming to assess the potential for cross-feeding in populations of motile bacteria. To this end, four bacteria genotypes are placed on a 2D lattice. Those bacteria will compete for survival until only one strain remains. Details of the strain are explained in the table below :

Strain name | evolutionary strategy | Secretions | Corresponding number in the code
--|--|--|--
Producers | Intra-genotype cooperation | `[1,1]` | `3`
Cross-feeders | Inter-genotype cooperation (and exploitation) | `[0,1]` and `[1,0]` | `1` and `2`
Defectors | Exploitation | `[0,0]` | `0`

For more information about the model, please read Section 2 of the report `Bacterial motility and the evolution of microbial cross-feeding.pdf`.

# How to use
## Parameters
The outcome of the simulation depends on many parameters, whose meaning and default values are detailed in the table below :

Variable name | Default value | Description
--|--|--
`L` | `100` | Size of the square lattice (so that there are `L*L` spaces).
`SELECTION_RATE` | `1.` | Rate of the selection reaction (see report).
`REPRODUCTION_RATE` | `1.` | Rate of the reproduction reaction (see report).
`DEATH_RATE` | `0.05` | Rate of the death reaction (see report).
`GOOD` | `0.1` | Fitness gained for each secretion produced in a bacteria's neighborhood.
`COST` | `0.05` | Fitness cost of each secretion produced by a bacteria.
`BASAL_FITNESS` | `0.1` | Basal fitness shared by all bacteria, without regard to the secretions.
`WEIGHT` | `0.5` | Necessity of the second secretion for the fitness to increase (if `WEIGHT == 0`), then both secretions are required.
`RADIUS` | `2` | Radius of the neighborhood of a bacteria, so that the neighborhood has `(2*RADIUS+1)**2` spaces.
`DETECTION_RADIUS` | `3` | Radius of the area around the bacteria taken into account for chemotaxis and density-dependant motility.
`MIN_EXCHANGE_RATE` | `None` | Turns on density-dependent motility, and sets the minimal exchange rate of bacteria where the local density is high enough.
`DD_THRESHOLD` | `0.5` | If `MIN_EXCHANGE_RATE is not None`, density threshold for the bacteria to slow down.
`INSULATORS_DENSITY` | `0` | Initial density of social insulators.
`CHEMOTAXIS` | `0` | Sets the bias of the bacteria's random walks towards the secretions. If `CHEMOTAXIS==0`, then all bacteria move randomly. If `CHEMOTAXIS==1`, all bacteria are attracted towards the secretions that they do not produce. If `CHEMOTAXIS==2`, they are attracted by the secretions by they produce. If `CHEMOTAXIS==3`, all bacteria are attracted by all secretions.
`BETA` | `0.2` | Legacy parameter which was used for an alternative implementation of chemotaxis. In this version it does not have any effect.
`BLOCKING` | `False` | Whether bacteria are allowed to carry out exchanges with spaces that are already occupied by another bacteria.
`SPAWN_FULL_PRODUCERS` | `True` | Whether producers `[1,1]` spawn at the beginning of the simulation. Used to study pairwise interactions between strains.
`SPAWN_FULL_DEFECTORS` | `True` | Whether defectors `[0,0]` spawn at the beginning of the simulation. Used to study pairwise interactions between strains.
`SPAWN_CROSS_FEEDERS` | `True` | Whether cross-feeders `[0,1]` and `[1,0]` spawn at the beginning of the simulation. Used to study pairwise interactions between strains.
`SAVE_HISTORY` | `False` | Whether to save the state of the lattice at each generation. If `SAVE_HISTORY`, only one replicate will be done per set of parameters.
`SAVE_LOG` | `False` | Whether to record the result of each interaction and exchange that have taken place during the simulation. ⚠️ The resulting file can be very heavy.

The default values of each parameter is stored inside `default_params.json`.

## Running simulations
To run simulations, run `main.py`, which will run `sim.py` for every combination of parameters.
To set the different parameters variation, change the values of the dictionnaries `dictRows`, `dictCols` and `generalParams` in `main.py`, in the `if name == "__main__"` section. `sim.py` will then be run with each combination of parameters described `dictCols` and `dictRows`, as well as with the general parameters described by `generalParams`. Parameters whose name do not appear in any of those dictionnary will have its default value used.
## Plotting 
Plotting the results of the simulations can be done with `plot3.py`. The code used to plot each figure in the report has been kept there for reproduction purposes.

Several values can be plotted with `plot3.py`, by using a different `dtype` key in the dictionnaries. These values are described in the following table:
`dtype` value | Data plotted
--|--
`0` | Extinction probability of each genotype.
`1` | Number of living cells at the generation when the second strain goes extinct, leaving only one alive.
`2` | Generation number of the extinction event described above.
`3` | Probability that all strains eventually go extinct.
`4` | Probability that a given genotype is the last one standing, without considering whether it will eventually go extinct.
`5` | Number of simulations ran for each value of the motility rate.
`6` | Number of living cell of each genotype at each generation (requires a history file and an `exchangeRate` key in the dictionnary).
`7` | Snapshot of the spatial structure at a given generation (requires a history file, and a `exchangeRate` and `gen` key.
`8` | Number death by selection event for each genotype and at each generation (requires a history file and an `exchangeRate` key in the dictionnary).
`9` | Number of successful reproduction event for each genotype and at each generation (requires a history file and an `exchangeRate` key in the dictionnary).
## Animations
Animations are done by using `animations.py`, and require an history file and a log file (which can be recorded by using `SAVE_HISTORY=True` and `SAVE_LOG=True`. 
Examples of such animations are provided in the corresponding folder.
