## Code to produce results from "Dynamic Modeling and Nonlinear Model Predictive Control..."
"Dynamic modeling and nonlinear model predictive control of a moving bed
chemical looping combustion reactor." Robert Parker and Lorenz Biegler.
Presented at DYCOPS 2022.

This code was originally run with IDAES 1.12 and Pyomo 6.2.
The files that produce the results are:
- `run_index_analysis.py`
- `run_stability_analysis.py`
- `run_nmpc.py`

All code should be run from this directory, as it relies on local
imports.
Subdirectories `common` and `mbclc` contain utility code and code to
construct and initialize the MBCLC model, respectively.

Because the stability analysis and NMPC scripts are time-consuming to
run, sample data is provided:
- `error_2_4_8_16.json` &mdash; Contains errors for the first four discretization
grids analyzed.
- `nmpc_input_data.json` &mdash; Contains control inputs from NMPC simulation
- `nmpc_plant_data.json` &mdash; Contains plant state variables from NMPC
simulation

Data may be plotted by running the `plot_discretization_errors.py`
and `plot_nmpc_trajectories.py` files.

If the methodology or code here has been useful in your research, please cite:
```
@article{parker2022,
author={Robert B. Parker and Lorenz T. Biegler},
title={Dynamic modeling and nonlinear model predictive control of a moving bed
chemical looping combustion reactor},
journal={IFAC PapersOnLine},
year={2022},
}
```
