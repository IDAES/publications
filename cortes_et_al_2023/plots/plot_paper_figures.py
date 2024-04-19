'''
This script reproduces plots from 'Market Optimization and Technoeconomic Analysis of Hydrogen-Electricity Coproduction Systems'
This excludes Figure 1 - process concept flow sheets and SI 6-7 (located in breakeven plots folder)
'''

from ies_model_library import *
from plot_signal_comparisons import * 
from linear_regression import * 
from iessolution import *

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
case5.plot_marginal_cost(fdest = 'figures/figure2boxaleft', fdest_h2= 'figures/figure2boxaright')

# Figure 2 Box B - histogram of LMP signal MiNg_$100_PJM-W_2035 

# create solution object 
soln = IESSolution(csv_file = '../market_results_20/MiNg_$100_PJM-W_2035_model5.csv', case_object = case5)

# generate plot 
soln.plot_lmpsignal_hist(signal_name = 'MiNg_$100_PJM-W_2035', fdest = 'figures/figure2boxb')


# Figure 2 Box C/Figure S8 A and B - SOFC + SOEC concept optimization results for 'MiNg_$100_MISO-W_2035' LMP scenario and $2/kg hydrogen (first 700 hours)
# Figure 2 Box A and Figure S8 A - net power output schedule
# Figure S8 B - hydrogen output schedule 

# create case object for this scenario 
soln = IESSolution(csv_file = '../market_results_20/MiNg_$100_MISO-W_2035_model5.csv', case_object = case5)

# generate the plot 
soln.plot_operation(start_end = (0,700), fdest = 'figures/figure2boxc', fdest_h2 = 'figures/figureS8b')

# Figure 2 Box D - Percentage of LMP scenarios with positive profits at $2/kg 
plot_positiveprofit_barplot(h2_price = 20, save = True)

# move to figures folder 
os.rename('positiveprofits_bar_20.png', 'figures/figure2boxd.png')
os.rename('positiveprofits_bar_20.pdf', 'figures/figure2boxd.pdf')

# Figure 3 - Annual proits by system at $2/kg hydrogen, sorted by medain
plt_signal_comparison('../market_results_20', PROFIT = True, CAPACITY = False, OUTPUT = False, sort = 'median', relative_case = None, save = True)

# move to figure folder 
os.rename('all_profit_20_median.png', 'figures/figure3.png')
os.rename('all_profit_20_median.pdf', 'figures/figure3.pdf')


# Figure 4 - scatter plot of annual profits, plotted against bimodality metrics for SOFC technologies 
plot_bimodalprofit_scatter(h2_price = 20, save = True)

# move to figures folder 
os.rename('bimodality_scatter_20.png', 'figures/figure4.png')
os.rename('bimodality_scatter_20.pdf', 'figures/figure4.pdf')

# Figure 5  - scatter plot of annual profits for gas price vs median LMP. Side by side plots at $1/kg , $2/kg, $3/kg 
# note - extra systems will be made, only plots included in figure 5 will be included here 
plot_profit_scatter(h2_price = 10, x = 'gasprice', y = 'medianlmp', side_by_side = True, save = True)
plot_profit_scatter(h2_price = 20, x = 'gasprice', y = 'medianlmp', side_by_side = True, save = True)
plot_profit_scatter(h2_price = 30, x = 'gasprice', y = 'medianlmp', side_by_side = True, save = True)

# rename plots from figure 5 - SOFC + SOEC and SOEC 
os.rename('SOFC+SOEC_gasprice_medianlmp_10.png', 'figures/figure5_topright.png')
os.rename('SOFC+SOEC_gasprice_medianlmp_20.png', 'figures/figure5_middleright.png')
os.rename('SOFC+SOEC_gasprice_medianlmp_30.png', 'figures/figure5_bottomright.png')

os.rename('SOEC_gasprice_medianlmp_10.png', 'figures/figure5_topleft.png')
os.rename('SOEC_gasprice_medianlmp_20.png', 'figures/figure5_middleleft.png')
os.rename('SOEC_gasprice_medianlmp_30.png', 'figures/figure5_bottomleft.png')

os.rename('SOFC+SOEC_gasprice_medianlmp_10.pdf', 'figures/figure5_topright.pdf')
os.rename('SOFC+SOEC_gasprice_medianlmp_20.pdf', 'figures/figure5_middleright.pdf')
os.rename('SOFC+SOEC_gasprice_medianlmp_30.pdf', 'figures/figure5_bottomright.pdf')

os.rename('SOEC_gasprice_medianlmp_10.pdf', 'figures/figure5_topleft.pdf')
os.rename('SOEC_gasprice_medianlmp_20.pdf', 'figures/figure5_middleleft.pdf')
os.rename('SOEC_gasprice_medianlmp_30.pdf', 'figures/figure5_bottomleft.pdf')

# move all other ones into folder 
os.rename('NGCC+SOEC_gasprice_medianlmp_10.png', 'figures/NGCC+SOEC_gasprice_medianlmp_10.png')
os.rename('NGCC+SOEC_gasprice_medianlmp_20.png', 'figures/NGCC+SOEC_gasprice_medianlmp_20.png')
os.rename('NGCC+SOEC_gasprice_medianlmp_30.png', 'figures/NGCC+SOEC_gasprice_medianlmp_30.png')

os.rename('NGCC_gasprice_medianlmp_10.png', 'figures/NGCC_gasprice_medianlmp_10.png')
os.rename('NGCC_gasprice_medianlmp_20.png', 'figures/NGCC_gasprice_medianlmp_20.png')
os.rename('NGCC_gasprice_medianlmp_30.png', 'figures/NGCC_gasprice_medianlmp_30.png')

os.rename('NGCC+SOEC_gasprice_medianlmp_10.pdf', 'figures/NGCC+SOEC_gasprice_medianlmp_10.pdf')
os.rename('NGCC+SOEC_gasprice_medianlmp_20.pdf', 'figures/NGCC+SOEC_gasprice_medianlmp_20.pdf')
os.rename('NGCC+SOEC_gasprice_medianlmp_30.pdf', 'figures/NGCC+SOEC_gasprice_medianlmp_30.pdf')

os.rename('NGCC_gasprice_medianlmp_10.pdf', 'figures/NGCC_gasprice_medianlmp_10.pdf')
os.rename('NGCC_gasprice_medianlmp_20.pdf', 'figures/NGCC_gasprice_medianlmp_20.pdf')
os.rename('NGCC_gasprice_medianlmp_30.pdf', 'figures/NGCC_gasprice_medianlmp_30.pdf')

os.rename('rSOC_gasprice_medianlmp_10.png', 'figures/rSOC_gasprice_medianlmp_10.png')
os.rename('rSOC_gasprice_medianlmp_20.png', 'figures/rSOC_gasprice_medianlmp_20.png')
os.rename('rSOC_gasprice_medianlmp_30.png', 'figures/rSOC_gasprice_medianlmp_30.png')

os.rename('SOFC_gasprice_medianlmp_10.png', 'figures/SOFC_gasprice_medianlmp_10.png')
os.rename('SOFC_gasprice_medianlmp_20.png', 'figures/SOFC_gasprice_medianlmp_20.png')
os.rename('SOFC_gasprice_medianlmp_30.png', 'figures/SOFC_gasprice_medianlmp_30.png')

os.rename('rSOC_gasprice_medianlmp_10.pdf', 'figures/rSOC_gasprice_medianlmp_10.pdf')
os.rename('rSOC_gasprice_medianlmp_20.pdf', 'figures/rSOC_gasprice_medianlmp_20.pdf')
os.rename('rSOC_gasprice_medianlmp_30.pdf', 'figures/rSOC_gasprice_medianlmp_30.pdf')

os.rename('SOFC_gasprice_medianlmp_10.pdf', 'figures/SOFC_gasprice_medianlmp_10.pdf')
os.rename('SOFC_gasprice_medianlmp_20.pdf', 'figures/SOFC_gasprice_medianlmp_20.pdf')
os.rename('SOFC_gasprice_medianlmp_30.pdf', 'figures/SOFC_gasprice_medianlmp_30.pdf')


# figure 6 - parity plots for lienar regression 
results = perform_regression_full()

plot_parityplots(results, save = True)

# move to figures folder 
os.rename('ols_parity.png', 'figures/figure6.png')
os.rename('ols_parity.pdf', 'figures/figure6.pdf')

# Figure 7 - Feature correlation 
corr_df = plot_featurecorrelation(results, save = True)

# move to results folder 
os.rename('feature_correlation_plot.png', 'figures/figure7.png')
os.rename('feature_correlation_plot.pdf', 'figures/figure7.pdf')
os.rename('feature_correlation.csv', 'figures/feature_correlation.csv')

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

# S9-10 - Capacity factors of all systems at $2/kg hydrogen 
plt_signal_comparison('../market_results_20', PROFIT = False, CAPACITY = True, OUTPUT = False, sort = 'median', relative_case = None, save = True)

# rename 
os.rename('all_p capacity_20_median.png', 'figures/figures9.png')
os.rename('all_p capacity_20_median.pdf', 'figures/figures9.pdf')

os.rename('all_h capacity_20_median.png', 'figures/figures10.png')
os.rename('all_h capacity_20_median.pdf', 'figures/figures10.pdf')

# S11-13 - Profit and capacity factors of all systems at $1/kg hydrogen 
plt_signal_comparison('../market_results_10', PROFIT = True, CAPACITY = True, OUTPUT = False, sort = 'median', relative_case = None, save = True)

# rename 
os.rename('all_profit_10_median.png', 'figures/figures11.png')
os.rename('all_profit_10_median.pdf', 'figures/figures11.pdf')

os.rename('all_p capacity_10_median.png', 'figures/figures12.png')
os.rename('all_p capacity_10_median.pdf', 'figures/figures12.pdf')

os.rename('all_h capacity_10_median.png', 'figures/figures13.png')
os.rename('all_h capacity_10_median.pdf', 'figures/figures13.pdf')

# S14-16 - Profit and capacity factors of all systems at $1.5/kg hydrogen 
plt_signal_comparison('../market_results_15', PROFIT = True, CAPACITY = True, OUTPUT = False, sort = 'median', relative_case = None, save = True)

# rename 
os.rename('all_profit_15_median.png', 'figures/figures14.png')
os.rename('all_profit_15_median.pdf', 'figures/figures14.pdf')

os.rename('all_p capacity_15_median.png', 'figures/figures15.png')
os.rename('all_p capacity_15_median.pdf', 'figures/figures15.pdf')

os.rename('all_h capacity_15_median.png', 'figures/figures16.png')
os.rename('all_h capacity_15_median.pdf', 'figures/figures16.pdf')

# S17-19 - Profit and capacity factors of all systems at $2.5/kg hydrogen 
plt_signal_comparison('../market_results_25', PROFIT = True, CAPACITY = True, OUTPUT = False, sort = 'median', relative_case = None, save = True)

# rename 
os.rename('all_profit_25_median.png', 'figures/figures17.png')
os.rename('all_profit_25_median.pdf', 'figures/figures17.pdf')

os.rename('all_p capacity_25_median.png', 'figures/figures18.png')
os.rename('all_p capacity_25_median.pdf', 'figures/figures18.pdf')

os.rename('all_h capacity_25_median.png', 'figures/figures19.png')
os.rename('all_h capacity_25_median.pdf', 'figures/figures19.pdf')

# S20-22 - Profit and capacity factors of all systems at $3/kg hydrogen 
plt_signal_comparison('../market_results_30', PROFIT = True, CAPACITY = True, OUTPUT = False, sort = 'median', relative_case = None, save = True)

# rename 
os.rename('all_profit_30_median.png', 'figures/figures20.png')
os.rename('all_profit_30_median.pdf', 'figures/figures20.pdf')

os.rename('all_p capacity_30_median.png', 'figures/figures21.png')
os.rename('all_p capacity_30_median.pdf', 'figures/figures21.pdf')

os.rename('all_h capacity_30_median.png', 'figures/figures22.png')
os.rename('all_h capacity_30_median.pdf', 'figures/figures22.pdf')

# S23-27 - System profit for all hydrogen prices, sorted chronologically 

# $1/kg
plt_signal_comparison('../market_results_10', PROFIT = True, CAPACITY = False, OUTPUT = False, sort = 'source', relative_case = None, save = True)

# move to figure folder 
os.rename('all_profit_10_source.png', 'figures/figures23.png')
os.rename('all_profit_10_source.pdf', 'figures/figures23.pdf')

# $1.5/kg
plt_signal_comparison('../market_results_15', PROFIT = True, CAPACITY = False, OUTPUT = False, sort = 'source', relative_case = None, save = True)

# move to figure folder 
os.rename('all_profit_15_source.png', 'figures/figures24.png')
os.rename('all_profit_15_source.pdf', 'figures/figures24.pdf')

# $2/kg 
plt_signal_comparison('../market_results_20', PROFIT = True, CAPACITY = False, OUTPUT = False, sort = 'source', relative_case = None, save = True)

# move to figure folder 
os.rename('all_profit_20_source.png', 'figures/figures25.png')
os.rename('all_profit_20_source.pdf', 'figures/figures25.pdf')

# $2.5/kg 
plt_signal_comparison('../market_results_25', PROFIT = True, CAPACITY = False, OUTPUT = False, sort = 'source', relative_case = None, save = True)

# move to figure folder 
os.rename('all_profit_25_source.png', 'figures/figures26.png')
os.rename('all_profit_25_source.pdf', 'figures/figures26.pdf')

# $3/kg 
plt_signal_comparison('../market_results_30', PROFIT = True, CAPACITY = False, OUTPUT = False, sort = 'source', relative_case = None, save = True)

# move to figure folder 
os.rename('all_profit_30_source.png', 'figures/figures27.png')
os.rename('all_profit_30_source.pdf', 'figures/figures27.pdf')

# S28 - profit scatter at $2/kg of dip test p-value and median LMP 
plot_profit_scatter(h2_price = 20, x = 'log10p', y = 'medianlmp', side_by_side = True, save = True)

# move to figure folder 
os.rename('NGCC_log10p_medianlmp_20.png', 'figures/figures28_topleft.png')
os.rename('NGCC_log10p_medianlmp_20.pdf', 'figures/figures28_topleft.pdf')

os.rename('SOFC_log10p_medianlmp_20.png', 'figures/figures28_bottomleft.png')
os.rename('SOFC_log10p_medianlmp_20.pdf', 'figures/figures28_bottomleft.pdf')

os.rename('NGCC+SOEC_log10p_medianlmp_20.png', 'figures/figures28_topright.png')
os.rename('NGCC+SOEC_log10p_medianlmp_20.pdf', 'figures/figures28_topright.pdf')

os.rename('rSOC_log10p_medianlmp_20.png', 'figures/rSOC_log10p_medianlmp_20.png')
os.rename('rSOC_log10p_medianlmp_20.pdf', 'figures/rSOC_log10p_medianlmp_20.pdf')

os.rename('SOFC+SOEC_log10p_medianlmp_20.png', 'figures/figures28_bottomright.png')
os.rename('SOFC+SOEC_log10p_medianlmp_20.pdf', 'figures/figures28_bottomright.pdf')

os.rename('SOEC_log10p_medianlmp_20.png', 'figures/SOEC_log10p_medianlmp_20.png')
os.rename('SOEC_log10p_medianlmp_20.pdf', 'figures/SOEC_log10p_medianlmp_20.pdf')

# S29-32 - percentage of scenarios with positive profits for hydogen prices $1, $1.5, $2.5, $3.0

# $1/kg
plot_positiveprofit_barplot(h2_price = 10, save = True)

# move to figures folder 
os.rename('positiveprofits_bar_10.png', 'figures/figures29.png')
os.rename('positiveprofits_bar_10.pdf', 'figures/figures29.pdf')

# $1.5/kg
plot_positiveprofit_barplot(h2_price = 15, save = True)

# move to figures folder 
os.rename('positiveprofits_bar_15.png', 'figures/figures30.png')
os.rename('positiveprofits_bar_15.pdf', 'figures/figures30.pdf')

# $2.5/kg
plot_positiveprofit_barplot(h2_price = 25, save = True)

# move to figures folder 
os.rename('positiveprofits_bar_25.png', 'figures/figures31.png')
os.rename('positiveprofits_bar_25.pdf', 'figures/figures31.pdf')

# $3/kg
plot_positiveprofit_barplot(h2_price = 30, save = True)

# move to figures folder 
os.rename('positiveprofits_bar_30.png', 'figures/figures32.png')
os.rename('positiveprofits_bar_30.pdf', 'figures/figures32.pdf')




