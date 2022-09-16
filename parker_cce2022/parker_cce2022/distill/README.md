## Distillation column case study
Contents:
- `distill_DAE.py` -- Code to build and instance of the distillation column model
- `distill_DAE.dat` -- .dat file for the distillation column model. This non-Python file should be included if this repository was cloned from GitHub.
- `run_full_space.py` -- Script to run the full space optimization problem
- `run_implicit_function.py` -- Script to run the implicit function optimization problem
- `compute_error.py` -- Script to compute difference between solutions in full space and implicit function optimization problems

IPOPT executable can be set by setting the `IPOPT_EXECUTABLE` environment variable, e.g.
```sh
export IPOPT_EXECUTABLE="/path/to/ipopt"
```
This should be done to make sure that the same IPOPT executable is used between
full space and implicit function solves.
