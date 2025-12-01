PZ/AFS Flowsheet Model
=======================
Disclaimer: This code was developed at Carnegie Mellon University by Ilayda Akkor and Chrysanthos E. Gounaris, as part of a research project funded by The Dow Chemical Company. In particular, we acknowledge the close collaboration with the following researchers from Dow: Shachit S. Iyer, John Dowdle and Le Wang, who provided key contributions pertaining to the project conceptualization, the research design and the methodology, model design choices and rationales, and the discussions of the results. 

If you find this code useful for your research, please consider citing
"Mathematical Modeling and Economic Optimization of a Piperazine-Based
Post-Combustion Carbon Capture Process" by Ilayda Akkor, Shachit S. Iyer,
John Dowdle, Le Wang and Chrysanthos E. Gounaris.

```
@article{akkor2025mathematical,
  title={Mathematical modeling and economic optimization of a piperazine-based post-combustion carbon capture process},
  author={Akkor, Ilayda and Iyer, Shachit S and Dowdle, John and Wang, Le and Gounaris, Chrysanthos E},
  journal={International Journal of Greenhouse Gas Control},
  volume={140},
  pages={104282},
  year={2025},
  publisher={Elsevier}
}
```

This is a model of the of the Piperazine/Advanced Flash Stripper (PZ/AFS) flowsheet [1],
which is a post-combustion carbon capture (PCC) process. The process consists of the following,

    * Absorber column (with an intercooler)
    * Stripper column connected to a flash tank (AFS)
    * Heat exchanger network (three exchangers)
    * Lean solvent cooler
    * Steam heater
    * Two bypasses
    * Recycle between columns with the addition of PZ and water make-up.

The column model is in the `column` directory in the file named `PZ_solvent_column.py`.
This model is connected to property models for the gas and liquid phases, which can
be found in the `properties` directory. The `Advanced_Flash_Stripper.py` file uses this column
model and connects it to the flash model located in `flash_tank.py`

The heat exchanger network model along with the heater and coolers are modeled in the
`PZ_AFS_heat_excahnger_network.py` file, where the heat duty and the area of all exchangers
are calculated.

The general flowsheet model is constructed in the `PZ_AFS_flowsheet.py` file. This file calls
the unit models from other files and defines the intermediate streams, bypasses and connectors.
Instructions on how to run this file for simulation or for different optimization cases is explained
further down.

More information about the model details can be found in the paper by Akkor et al. [2]

Environment
------------------
The dependencies required to run this code has been frozen in the
`requirements.txt` file, and can be installed with
```console
pip install -r requirements.txt
```
Python 3.8.10 was used.

CONOPT version 3.17N was used as the solver accessed through GAMS (v 45.6.0)


Simulation and Optimization Cases
------------------
This flowsheet model can be run for seven different cases, concerning the plant scale,
flue gas type and plant configuration.

    * Case 0 - simulation of the pilot scale, coal-fired flue gas processing plant
    * Case 1 - optimization of the pilot scale, coal-fired flue gas processing plant
    * Case 2 - optimization of the pilot scale, NGCC flue gas processing plant
    * Case 3 - optimization of the commercial scale, coal-fired flue gas processing plant
    * Case 3* - optimization of the commercial scale, coal-fired flue gas processing plant in split train configuration
    * Case 4 - optimization of the commercial scale, NGCC flue gas processing plant
    * Case 4* - optimization of the commercial scale, NGCC flue gas processing plant in split train configuration
as shown in the paper by Akkor et al [2].

The code can be run with the following,


`python PZ_AFS_flowsheet.py  <flue_gas> <scale> <optimization> <split_train>`

where

`<flue_gas>` can be set to `'pilot'` (processing 20 mol/s of flue gas) or
`'commercial'` (processing 30,000 mol/s of flue gas)

`<scale>` can be set to `'coal'` for coal-fired flue gas (contains 10.5%mol CO2) or
to `'NGCC'` for natural gas combined cycle flue gas (containing 4.1%mol CO2)

`<optimization>` can be set to `False` to do a simulation or to `True` which
will call the optimization scripts for the relevant case (where an economic objective is introduced
and the degrees of freedom are freed gradually)

`<split_train>` can be set to `True` for two parallel trains of the process each processing 
half the amount of flue gas or to `False` for a single train


Custom Runs
------------------
The model has been tested extensively with the conditions outlined above.
For a custom run with different specifications, the user can change the values
of any of the design variables or inlet values (e.g. flue gas amount and composition)
that are fixed in the `PZ_AFS_flowsheet.py`, however we offer no convergence 
guarantee. From prior experience, we suggest that the code to be run with the
case that is closest to the custom run desired, and then that solution must be 
used as the initialized model for the custom run. Next, the model can be solved
iteratively until the desired specification is reached. For example, if the desired 
run is to optimize a plant processing 28,000 mol/s of NGCC flue gas, best course of action 
is to run the code for commercial scale and add the following steps at the end of the code,

    m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(29000)
    results = solver.solve(m.fs)
    print(results.solver.termination_condition)

    m.fs.absorber.vapor_properties[0, 0].flow_mol.fix(28000)
    results = solver.solve(m.fs)
    print(results.solver.termination_condition)


Moreover, the user has the option to change the number of finite elements and to
add or remove the intercooler and the linear pressure drop for the columns.


Degrees of Freedom
------------------

The process model has ten design and two process variables.

    * Packed length of absorber column
    * Inner diameter of absorber column
    * Packed length of stripper column
    * Inner diameter of stripper column
    * Intercooler area
    * Lean cooler area
    * Steam heater area
    * Exchanger areas (three)
    * Bypass ratios (two)
    * Solvent flowrate

References
------------

1. GT Rochelle, Y Wu, E Chen, K Akinpelumi, KB
Fischer, T Gao, CT Liu, JL Selinger. Pilot plant
demonstration of piperazine with the advanced flash
stripper. Int J Greenh Gas Control 84:72-81 (2019)

2. I Akkor, SS Iyer, J Dowdle, L Wang, CE Gounaris.
Mathematical modeling and economic optimization of a
piperazine-based post-combustion carbon capture
process. (To appear, 2024)
