### Code to produce the results from "An application of the Dulmage-Mendelsohn partition..."
"An application of the Dulmage-Mendelsohn partition to the analysis of a discretized dynamic chemical looping reactor model." Robert Parker, Chinedu Okoli, Bethany Nicholson, John Siirola, and Lorenz Biegler. To be presented at FOCAPO 2023.

The code is structured as a Python package with a small `setup.py` so that functions and classes may be more easily tested and imported by external scripts. Please run
```console
python setup.py develop
```
to install the `parker-focapo2023` package.

### Scripts that produce the paper results
- *partition_by_time.py* - Partitions the Jacobian by time (i.e. generates the left half of Figure 1) and checks the structural rank of subsystems at each point in time, as well as the full Jacobian
- *partition_by_dae.py* - Partitions the Jacobian at t=30 s into differential, algebraic, and discretization equations (and corresponding variables) by naively looking for Pyomo.DAE components. Note that the resulting partition does not yield a square algebraic subsystem.
- *partition_by_dae_valid.py* - Partitions the Jacoabian at t=30 s into differential, algebraic, and discretization subsystems with additional checking to make sure that differential variables are not included at the inlets (where they are fully specified by inputs and disturbances). The plot generated is the right half of Figure 2.
- *analyze_1_7_alg_jac.py* - Performs the Dulmage-Mendelsohn partition on the algebraic Jacobian at t=30 s, prints the variables and constraints that appear in the under and overconstrained systems, and displays these incidence matrices. This is Figure 2.
- *analyze_patched_system.py* - Checks the patched model for structural singularity, then applies the Dulmage-Mendelsohn partition to the algebraic Jacobian at t=30 s to make sure the under and over constrained subsystems are empty. Checks this system for numerical singularity. Generates the plots in Figure 3.
- *compare_dyn_opt.py* - Compares characteristics of a dynamic optimization problem between the two model versions. By default, only the KKT matrix is compared and the optimization problem is not solved (because it takes a long time for the solve with the 1.7 version to terminate). This can be changed by setting the `solve` variable to True in function `main`. These results form Table 1.
