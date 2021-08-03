# Model Readme

This summarizes the plant model contents. These files were used for
parameter estimation, validation, simulation, and optimization of the Escalante
Generating Station in Prewitt, NM.

Most of the equipment of the models included here were subsequently released as
part of IDAES. The released versions are improved, tested, and documented. It is
recommended that the latest official IDAES versions of the models be used in
developing new power plant models. The models included here are intended for
reference and documentation of parameter estimation and optimization results.

## Version Requirements

The models here are intended to be run with the 1.8.0 version of IDAES. They
will not run properly with other versions.

## Data

Operating data for Escalante Station were provided by Tri-State Generation
and Transmission Association.

## Contents

#### Directories

1) unit_models - contains equipment models for the power plant, updated versions
   of these were released and documented as part of the IDAES. The models
   included here were used to complete this work before the official release.

2) flowsheets - contains flowsheet models for the boiler and steam cycle
   subsections

3) data - this directory contains plant data

     a) data.csv - full plant data set

     b) data_small.csv - steady state data with unused columns removed

     c) test_set.csv - steady state time stamps

     d) meta_data.csv - meta data for columns, including mapping data tags to
                        the model

4) results - results files are stored in this directory

5) pfd - contains process flow diagram templates.  Model results can be written
   onto these diagrams.

6) tmp - temporary files

7) tmp_plot - temporary plots that to be combined into one file

#### Files

1) plot_dyn.py - combines the boiler and steam cycle flowsheet model into a full
   plant model.  Although the plant model is dynamic, only the steady state
   version is used for this work.

2) plant_steady.py - run the steady state model before parameter estimation

3) pest_big.py - run the parameter estimation problem.  This generates
   results/est.csv, which contains model parameters used by the validation and
   optimization scripts.

4) plant_validate.py - run validation on just the points used for parameter
   estimation.  This reads the results/est.csv file, and it stores the end
   result of each validation simulation as results/state4_{index}.json.gz. The
   stored state of the validation is also the read by pest_big.py and used as
   the initial guess, allowing for the parameter estimation to be improved
   incrementally if deficiencies are observed in the validation results.

5) make_box_plot_book.py - makes a box and whisker plot of the steady state data
   set.  It also plots the points used in the parameter estimation and the
   validation result on the box plots, so it requires the results of
   plant_validate.py

6) plant_validate_all.py - runs validation for all the steady state data points

7) plant_steady_slide.py - runs the baseline model cases using correlations for
   operating variables.  This represents typical current operation of the plant.
   This reads model parameters from results/est.csv.

8) plant_steady_opt.py - runs optimization cases. This reads model parameters
   from results/est.csv.  This file can be edited to enforced different
   constraints or used different objective or degrees of freedom.

9) plot_val.py - makes parity plots from just points used in parameter
   estimation.

10) plot_val_all.py - makes parity plots for all steady-state data.

11) correlation_and_optimzation_plots.xlsx - assemble validation, data, and
    optimization results, for plots
