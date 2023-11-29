## Code to produce results of "Dynamic Modeling and NMPC..."

Parker, Nicholson, Siirola, and Biegler. Submitted to J. Process Control.

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
$ pip install -r requirements.txt
```
IPOPT 3.13.2 was used for all square and optimization solves.

### Citation
If you use this code or the underlying `pyomo.contrib.mpc` software in
your research, please cite the paper:
```bibtex
@article{parker2023mpc,
title = {Model predictive control simulations with block-hierarchical differential-algebraic process models},
journal = {Journal of Process Control},
volume = {132},
pages = {103113},
year = {2023},
issn = {0959-1524},
doi = {https://doi.org/10.1016/j.jprocont.2023.103113},
url = {https://www.sciencedirect.com/science/article/pii/S0959152423002007},
author = {Robert B. Parker and Bethany L. Nicholson and John D. Siirola and Lorenz T. Biegler},
}
```
