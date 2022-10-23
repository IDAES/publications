### Code to produce the results from "An application of the Dulmage-Mendelsohn partition..."
"An application of the Dulmage-Mendelsohn partition to the analysis of a discretized dynamic chemical looping reactor model." Robert Parker, Chinedu Okoli, Bethany Nicholson, John Siirola, and Lorenz Biegler. To be presented at FOCAPO 2023.

The code is structured as a Python package with a small `setup.py` so that functions and classes may be more easily tested and imported by external scripts. Please run
```console
python setup.py develop
```
to install the `parker-focapo2023` package. Python dependencies can be found in the
`requirements.txt` file, and can be installed with
```console
pip install -r requirements.txt
```
This file describes the environment with which this code was produced and tested,
not necessarily the only environment with which it works.
Non-Python dependencies include Ipopt (tested with versions 3.13 and 3.14) and
MA27 (tested with CoinHSL 2.2.1).

### Organization
The code to actually produce the paper results is in the `parker_focapo2023/clc`
subdirectory. The `parker_focapo2023/mpc` and `parker_focapo2023/common`
subdirectories contain utility code for setting up a dynamic optimization problem
and partitioning the DAE model.

Within `parker_focapo2023/clc`, the three "gas solid contactors" directories
contain copied-and-pasted unit and property models from three different versions
of IDAES that were used testing and demonstrating the incidence graph analysis
code.

### Scripts that produce the paper results
- *clc/partition_by_time.py* - Partitions the Jacobian by time (i.e. generates the left half of Figure 1) and checks the structural rank of subsystems at each point in time, as well as the full Jacobian
- *clc/partition_by_dae.py* - Partitions the Jacobian at t=30 s into differential, algebraic, and discretization equations (and corresponding variables) by naively looking for Pyomo.DAE components. Note that the resulting partition does not yield a square algebraic subsystem.
- *clc/partition_by_dae_valid.py* - Partitions the Jacoabian at t=30 s into differential, algebraic, and discretization subsystems with additional checking to make sure that differential variables are not included at the inlets (where they are fully specified by inputs and disturbances). The plot generated is the right half of Figure 2.
- *clc/analyze_1_7_alg_jac.py* - Performs the Dulmage-Mendelsohn partition on the algebraic Jacobian at t=30 s, prints the variables and constraints that appear in the under and overconstrained systems, and displays these incidence matrices. This is Figure 2.
- *clc/analyze_patched_system.py* - Checks the patched model for structural singularity, then applies the Dulmage-Mendelsohn partition to the algebraic Jacobian at t=30 s to make sure the under and over constrained subsystems are empty. Checks this system for numerical singularity. Generates the plots in Figure 3.
- *clc/compare_dyn_opt.py* - Compares characteristics of a dynamic optimization problem between the two model versions. By default, only the KKT matrix is compared and the optimization problem is not solved (because it takes a long time for the solve with the 1.7 version to terminate). This can be changed by setting the `solve` variable to True in function `main`. These results form Table 1.

### Tests

Please run
```console
pytest
```
to ensure that the code and dependencies are properly installed.
This runs tests for the dynamic optimization dependencies, the unit and property
models for three different versions of IDAES, and the decomposition and
analysis itself.

### Citation
