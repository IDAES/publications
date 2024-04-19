PZ Solvent Column
====================

Disclaimer: This code was developed at Carnegie Mellon University by Ilayda Akkor and Chrysanthos E. Gounaris,
as part of a research project funded by The Dow Chemical Company. In particular, we acknowledge the close collaboration
with the following researchers from Dow: Shachit S. Iyer, John Dowdle and Le Wang, who provided key contributions pertaining
to the project conceptualization, the research design and the methodology, model design choices and rationales, and the discussions of the results.

If you find this code useful for your research, please consider citing
"Mathematical Modeling and Economic Optimization of a Piperazine-Based
Post-Combustion Carbon Capture Process" by Ilayda Akkor, Shachit S. Iyer,
John Dowdle, Le Wang and Chrysanthos E. Gounaris

This column model is to be used as the absorber and stripper columns as part
of the Piperazine/Advanced Flash Stripper (PZ/AFS) process [1],
which is a post-combustion carbon capture (PCC) process. For the absorber case,
flue gas containing CO2 enters the packed column from the bottom and is contacted with the
amine solvent counter-currently. For the stripper column, vapor obtained after flashing the rich
solvent enters the column from the bottom and contacts the rich solvent coming from
the top. It is a rate-based column model that uses the two-film theory for material
and energy transfer and accounts for kinetics by using an enhancement factor. There is an option
to add a in-and-out type intercooler in the middle of the column and
to add a linear pressure drop with a fixed outlet, which can be set via configuration arguments
when instantiating the model.

A finite difference discretization is used where 40 segments are used (mutable).
A forward finite difference scheme is used for the liquid phase while a backwards
scheme is used for the vapor phase.

More information about the model details can be found in the paper by Akkor et al. [2]

Degrees of Freedom
------------------

The solvent column model has 2 design variables an 3 design parameters related to packing.

Design Variables:

    * Packed length
    * Inner diameter
    * Packing specific area (mutable parameter),
    * Packing void fraction (mutable parameter),
    * Packing corrugation angle (mutable parameter)

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
