## Code to produce results of "Dynamic Modeling and NMPC..."

Parker, Nicholson, Siirola, and Biegler. Submitted to *J. Process Control*.

Organization of the code:
- `parker_jpc2023` is a small Python package that can be installed with:
```
$ python setup.py develop
```
- `parker_jpc2023/examples` contains small examples that are presented in code
listings in Sections 3 and 4 of the paper
- `parker_jpc2023/clc` contains the code necessary to run the NMPC example
presented in Section 6 of the paper

### Dependencies
The results and code listings presented in the paper were prepared using
Pyomo 6.5.1 and IDAES 2.0.0. The full Python environment used can be found
in `requirements.txt` and can be installed with:
```
$ pipe install -r requirements.txt
```
IPOPT 3.13.2 was used for all square and optimization solves.
