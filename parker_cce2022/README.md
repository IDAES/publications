This directory contains code necessary to reproduce the results in the paper "An implicit function formulation for optimization of discretized index-1 differential algebraic systems"

### Environment
The environment in which this code was tested has been frozen in the
`requirements.txt` file, and can be installed with
```consolve
pip install -r requirements.txt
```
Python 3.9.2 was used.
Note the following major dependencies:
- IDAES 1.13.0
- Pyomo 6.4.1
- CyIpopt 1.1.0
CyIpopt is linked to IPOPT 3.12, compiled locally according to the instructions
found here: `https://coin-or.github.io/Ipopt/INSTALL.html`.

### Installation
The code is structured as a small Python package to make it easier to test this
code and manage imports. An added benefit is that this code may be easily imported
elsewhere for additional experimentation. Please run
```consolve
python setup.py develop
```
to "install" the `parker_cce2022` "package."

### Testing
To make sure the dependencies are properly installed, please run
```console
pytest
```
in this directory. This runs tests that solve a nominal instance of each
case study and check that the answer is as expected. They take a couple
of minutes to run.

### Structure
There are several subdirectories:
- `common` -- Utility code
- `mbclc` -- Code to construct instances of the Moving Bed CLC model
- `distill` -- Code to run the distillation column case study
- `mbclc_sim` -- Code to run the CLC simulation case study
- `mbclc_dynopt` -- Code to run the CLC dynamic optimization case studies
See the READMEs in each of the case study directories for instructions
on how to produce the results.
