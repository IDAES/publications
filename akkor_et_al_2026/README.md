Robust Optimization of the PZ/AFS Process Model
=======================

This directory contains the code used to produce the results in the paper "Epistemic Uncertainty Analysis and Robust Optimization of a Second-Generation Solvent-Based Post-Combustion Carbon Capture Process"

### Disclaimer
This code was developed at Carnegie Mellon University by Ilayda Akkor and Chrysanthos E. Gounaris, 
as part of a research project funded by The Dow Chemical Company. In particular, we acknowledge the close 
collaboration with the following researchers from Dow: Shachit S. Iyer, John Dowdle and Le Wang, 
who provided key contributions pertaining to the project conceptualization, the research design 
and the methodology, model design choices and rationales, and the discussions of the results. 

### Citation
If you find this code useful for your research, please consider citing "Epistemic Uncertainty Analysis and Robust Optimization of a Second-Generation Solvent-Based Post-Combustion Carbon Capture Process" by Ilayda Akkor, Shachit S. Iyer, John Dowdle, Le Wang and Chrysanthos E. Gounaris

### Dependencies
The dependencies required to run this code are in the
`requirements.txt` file, and can be installed with
```console
pip install -r requirements.txt
```

Python 3.9.23 was used.

Note the following major dependencies:
- Pyomo 6.9.1
- IDAES 2.8.0

Solvers:
- PyROS 1.3.5
- Local solvers (CONOPT3, CONOPT4, IPOPT, KNITRO) accessed through GAMS (v 48.6.1)

### Structure

The open-source PZ/AFS process model was introduced in former [work](https://www.sciencedirect.com/science/article/pii/S1750583624002251) and is available in a separate [directory](https://github.com/IDAES/publications/tree/main/akkor_et_al_2024). The model files (`PZ_AFS_flowsheet.py`, `PZ_solvent_column.py`, `Advanced_Flash_Stripper.py`, `flash_tank.py`, `PZ_AFS_heat_exchanger_network`, `economic_model.py`, `properties/.`) are included in this directory for completion.

`optimize_nominal_case.py` file is the deterministic optimization script used to initialize PyROS, variables are adjusted for the two-stage setting, as discussed in the manuscript.

`ro_script.py` is the main file to produce the results from the manuscript.

### Robust Optimization Script
This script demonstrates
- Set-up of the solver
- Problem initialization routine
- Uncertain parameters and the uncertainty ranges used
- Uncertainty set construction
- Arguments used for PyROS

This script contains all the different uncertainty sets shown in the manuscript. As an example, the current solver set-up is for solving the ellipsoid set concerning uncertainty in the Henry's constant of CO2, at a 95% confidence interval (Fig 7 in the manuscript).

