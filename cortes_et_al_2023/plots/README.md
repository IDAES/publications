# Plots 

This folder contains all files needed to reproduce the figures from the manuscript text.   
Here are descriptions of all the files.  

- `formatted_raw_data.csv`: .csv file containing raw data about each LMP signal including signal statistics, hydrogen price, natural gas price, and optimal annual profit results.  
- `ies_model_library.py`: Contains objects for each system concept with their surrogate equations. The parent class contains a function to replicate marginal cost plots from the manuscript.  
- `iessolution.py`: Contains code to save model solutions as an object and use the data to create plots and .csv files. This file contains code to plot histograms of the LMP signals as well as plots of the dispatch schedules for individual model solutions.   
- `plot_signal_comparisons.py`: Contains functions needed to plot comparisons between all cases. This includes Figure's 3-5 from the main text of the manuscript.  
- `linear_regression.py`: Contains functions needed to perform linear regression and generate plots (parity plots, feature correlation).    
- 'plot_paper_figures.py`: A script that reproduces each figure from the main text and SI of the manuscript. All images are saved in the 'figures' folder and saved as their figure number.  
- `plot_examples.ipynb`: Notebook containing examples of each plot and details on how to generate them.  
