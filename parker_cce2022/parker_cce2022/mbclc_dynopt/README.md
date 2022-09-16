## CLC dynamic optimization case study
Contents:
- `model.py` -- Code to construct the dynamic optimization problem
- `run_full_space.py` -- Script to run a full space optimization problem
- `run_implicit_function.py` -- Script to run an implicit function optimization problem
- `full_space_sweep.py` -- Code to set up a parameter sweep for the full space formulation
- `run_full_space_sweep_eqcon.py` -- Script to run a parameter sweep on the equality-constrained full space optimization problem
- `run_full_space_sweep_boundcon.py` -- Script to run a parameter sweep on the bound-constrained full space optimization problem
- `implicit_sweep.py` -- Script to set up a parameter sweep for the implicit function formulation
- `run_implicit_sweep_eqcon.py` -- Script to run a parameter sweep on the equality-constrained implicit function optimization problem
- `run_implicit_sweep_boundcon.py` -- Script to run a parameter sweep on the bound-constrained implicit function optimization problem
- `full_space_sweep_boundcon.json`, `full_space_sweep_eqcon.json`, `implicit_sweep_boundcon.json`, `implicit_sweep_eqcon.json` -- json files containing sample data from the above `run*sweep*` files, which are provided as the parameter sweeps take several hours to run.
- `analyze_data.py` -- Script to analyze and plot results of parameter sweeps. Relies on the above json files.
- `series_data.py`, `solve_data.py`, `sweep_data.py` -- Data structures used to communicate results of solves and parameter sweeps.

IPOPT executable can be set by setting the `IPOPT_EXECUTABLE` environment variable, e.g.
```sh
export IPOPT_EXECUTABLE="/path/to/ipopt"
```
This should be done to make sure that the same IPOPT executable is used between
full space and implicit function solves.
