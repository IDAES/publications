'''
This script reproduces plots from 'Market Optimization and Technoeconomic Analysis of Hydrogen-Electricity Coproduction Systems'
This excludes Figure 1 - process concept flow sheets and SI 6-7 (located in breakeven plots folder)
'''

from ies_model_library import *
from plot_signal_comparisons import * 
from linear_regression import * 
from iessolution import *

import os.path

# creat figures folder 

# Get the current directory
current_directory = os.getcwd()

# Define the folder name
folder_name = 'figures'

# Check if the folder exists
if not os.path.exists(os.path.join(current_directory, folder_name)):
    # Create the folder if it doesn't exist
    os.mkdir(os.path.join(current_directory, folder_name))


#################################################################################################################
## MAIN TEXT FIGURES 

# Figure 2 Box A - Marginal cost of SOFC + SOEC concept 

# create case object for case 5 
case5 = Case5() 

# return marginal cost plots 
case5.plot_marginal_cost(fdest='figures/figure2boxaleft', fdest_h2='figures/figure2boxaright')

# Figure 2 Box B - histogram of LMP signal MiNg_$100_PJM-W_2035 

# create solution object 
soln = IESSolution(csv_file = '../market_results_20/MiNg_$100_PJM-W_2035_model5.csv', case_object=case5)

# generate plot 
soln.plot_lmpsignal_hist(signal_name = 'MiNg_$100_PJM-W_2035', fdest = 'figures/figure2boxb')


# Figure 2 Box C/Figure S8 A and B - SOFC + SOEC concept optimization results for 'MiNg_$100_MISO-W_2035' LMP scenario and $2/kg hydrogen (first 700 hours)
# Figure 2 Box A and Figure S8 A - net power output schedule
# Figure S8 B - hydrogen output schedule 

# create case object for this scenario 
soln = IESSolution(csv_file='../market_results_20/MiNg_$100_MISO-W_2035_model5.csv', case_object=case5)

# generate the plot 
soln.plot_operation(start_end = (0,700), fdest = 'figures/figure2boxc', fdest_h2 = 'figures/figureS8b')

# Figure 2 Box D - Percentage of LMP scenarios with positive profits at $2/kg 
plot_positiveprofit_barplot(h2_price=20, save=True)

# move to figures folder 
os.rename('positiveprofits_bar_20.png', 'figures/figure2boxd.png')
os.rename('positiveprofits_bar_20.pdf', 'figures/figure2boxd.pdf')

# figure 6 - parity plots for lienar regression 
file_name = os.path.join("C:\\", "Users", "Dan", "Documents", "Python Scripts", "Joule Paper Figures", "Nicole_Paper_Figures", 
                         "src", "formatted_raw_data_DJL.csv")
results = perform_regression_full(fdest_data=file_name)

plot_parityplots(results, save=True)

# move to figures folder 
os.rename('ols_parity.png', 'figures/figure6.png')
os.rename('ols_parity.pdf', 'figures/figure6.pdf')

# Figure 7 - Feature correlation 
corr_df = plot_featurecorrelation(fdest_data=file_name, save=True)

# move to results folder 
os.rename('feature_correlation_plot_DJL.png', 'figures/figure7.png')
os.rename('feature_correlation_plot_DJL.pdf', 'figures/figure7.pdf')
os.rename('feature_correlation_DJL.csv', 'figures/feature_correlation.csv')

####################################################################################
## SI FIGURES

# S1 - Marginal Cost of power from NGCC system 
case0 = Case0() 

case0.plot_marginal_cost(fdest = 'figures/figures1')

# S2 - Marginal cost of power for SOFC system/rSOC in power mode

case1 = Case1()

case1.plot_marginal_cost(fdest = 'figures/figures2')

# S3 - Marginal cost of hydrogen for standalone SOEC/rSOC in hydrogen mode

case4 = Case4()

case4.plot_marginal_cost(fdest_h2 = 'figures/figures3')

# S4-5 - Marginal cost of power and hydrogen for NGCC + SOEC

case3 = Case3() 

case3.plot_marginal_cost(fdest = 'figures/figures4', fdest_h2 = 'figures/figures5')



