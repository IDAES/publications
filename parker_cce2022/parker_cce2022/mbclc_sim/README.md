## CLC simulation case study
Contents:
- `full_space.py` -- Script to run the full space simulation problem
- `reduced_space.py` -- Script to run the implicit function (reduced space) simulation problem
- `run_param_sweep.py` -- Script to run the parameter sweep presented in the paper. Produces a file named `param_sweep.json`.
- `param_sweep.json` -- Sample data from `run_param_sweep.py`. Provided as this script may take about an hour to run.
- `plot_grid.py` -- Script to plot the comparison of convergence status between the two formulations. Expects to find the `param_sweep.json` file _in the current working directory_.

IPOPT executable can be set by setting the `IPOPT_EXECUTABLE` environment variable, e.g.
```sh
export IPOPT_EXECUTABLE="/path/to/ipopt"
```
This should be done to make sure that the same IPOPT executable is used between
full space and implicit function solves.
