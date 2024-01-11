'''
This file contains all functions needed to perform linear regression 
using data from 'formatted_raw_data.csv' (found in this directory)
and plot regressions results/ produce tables 
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def read_raw_data(file_name):
    '''
    Reads csv file of raw data and returns pd.DataFrame

    Arguments:
        file_name: string, directory and file name of raw data csv 

    Returns:
        formatted_data: pd.DataFrame, dataframe containing the raw data 
    '''

    formatted_data = pd.read_csv('formatted_raw_data.csv')
    formatted_data.set_index('Unnamed: 0', inplace = True)

    return formatted_data 


def standardize_variables(df, inplace=True):
    '''
    Standardize a dataframe of individual variables based on their means and standard deviations 

    Arguments:
        df: pd.DataFrame, variables to be standardized, each column will be standardized individually 
        inplace: boolean, whether or not to create a new dataframe of standardized variables or modify the existing one 
            default = True, modify existing dataframe 

    Returns:
        None or pd.DataFrame, depending on inplace parameter
    '''
    if inplace:
        # loop through each column 
        for col in df.columns:
            # return the column as a np.array 
            a = df[col].to_numpy()

            # calculate the mean 
            mean = np.mean(a)

            # calculate the sd
            sd = np.std(a)

            # save the values as the standardized 
            for idx in df.index:
                df.at[idx, col] = (df.at[idx, col] - mean)/sd
        
        return None
    
    else:
        # create a copy of the dataframe
        df_std = df.copy()

        # loop through each column 
        for col in df_std.columns:
            # return the column as a np.array 
            a = df_std[col].to_numpy()

            # calculate the mean 
            mean = np.mean(a)

            # calculate the sd
            sd = np.std(a)

            # save the values as the standardized 
            for idx in df_std.index:
                df_std.at[idx, col] = (df_std.at[idx, col] - mean)/sd
        
        return df_std
    
def separate_profits_features(df):
    '''
    Separate out profits and features into two separate dataframes 

    Arguments:
        df: pd.DataFrame, dataframe of all the linear regression data 

    Returns:
        features: pd.DataFrame, dataframe of features from the raw data file 
        all_profits: pd.DataFrame, dataframe of profits from the raw data file
    '''

     # create a single feature matrix 
    features = df.drop(labels = ['NGCC profit (M$)','SOFC profit (M$)', 'NGCC+SOEC profit (M$)', 'rSOC profit (M$)', 'SOFC+SOEC profit (M$)', 'SOEC profit (M$)', 'Dip Test p-value'], axis = 1)

    # create a dataframe of all the profit columns 
    all_profits = df.drop(df.columns.difference(['NGCC profit (M$)','SOFC profit (M$)', \
                                                                         'NGCC+SOEC profit (M$)', 'rSOC profit (M$)', \
                                                                            'SOFC+SOEC profit (M$)', 'SOEC profit (M$)']), \
                                                                                axis = 1)

    return features, all_profits
                            

def standardize_profits(df):
    '''
    Standardizes all system profits as one group 

    Arguments:
        df: pd.DataFrame, single dataframe of all profit objects
        
    Returns:
        cases: dictionary, annual profit values, standardized in separate lists 
    '''

    # find the mean of all the NPVs 
    mean_value = df.values.mean()

    # find standard deviation
    sd = df.values.std()

    # create dataframe for each system's profit separately
    NGCC_profit = df.drop(df.columns.difference(['NGCC profit (M$)']), axis = 1)
    SOFC_profit = df.drop(df.columns.difference(['SOFC profit (M$)']), axis = 1)
    NGCCSOEC_profit = df.drop(df.columns.difference(['NGCC+SOEC profit (M$)']), axis = 1)
    rSOC_profit = df.drop(df.columns.difference(['rSOC profit (M$)']), axis = 1)
    SOFCSOEC_profit = df.drop(df.columns.difference(['SOFC+SOEC profit (M$)']), axis = 1)
    SOEC_profit = df.drop(df.columns.difference(['SOEC profit (M$)']), axis = 1)


    # loop through all the data frames and standardize using the overall stats
    for c in [NGCC_profit, SOFC_profit, NGCCSOEC_profit, rSOC_profit, SOFCSOEC_profit, SOEC_profit]:
        # return column of values as an array
        for idx in c.index:
            c.at[idx, c.columns[0]] = (c.at[idx, c.columns[0]] - mean_value)/sd

    cases = {'NGCC': NGCC_profit, 'SOFC': SOFC_profit, 'NGCC+SOEC': NGCCSOEC_profit, 'rSOC':rSOC_profit, 'SOFC+SOEC': SOFCSOEC_profit, 'SOEC': SOEC_profit}

    return cases


def perform_linear_regression(features, y, verbose = True):
    '''
    Performs linear regression given a set of standardized features and y values 
    uses OLS methods from statsmodels.api package

    Arguments:
        features: pd.DataFrame, standardized dataframe of features 
        y: pd.DataFrame, standardized dataframe of y values
        verbose: boolean, whether or not to print results 
            default = True, print results 

    returns:
        res: results object 
    '''

    # add constant
    features_ = sm.add_constant(features)

    # convert y to numpy array with data type float
    y_ = y.to_numpy(dtype = float)

    # OLS linear regression 
    ols = sm.OLS(y_, features_)

    # fit the data 
    res = ols.fit()

    if verbose:
        print(res.summary())

    return res

def perform_regression_full(fdest_data = 'formatted_raw_data.csv'):
    '''
    Performs OLS linear regression for each of the system concept annual profits in 
    'formatted_raw_data.csv'

    Arguments:
        fdest_data: string, destination of data file 
            default = 'formatted_raw_data.csv', located in this directory

    Returns:
        results: dict, results objects for each of the system concepts 
            keys = system concepts, values= results objects
    '''

    # read data file 
    formatted_data = read_raw_data(fdest_data)

    # separate profits and features 
    features, all_profits = separate_profits_features(formatted_data)

    # standardize features matrix 
    standardize_variables(features)

    cases = standardize_profits(all_profits)

    results = dict.fromkeys(cases.keys())
    
    # run the linear regression 
    for key in results.keys():
        results[key] = perform_linear_regression(features, cases[key])

    return results 

def return_coefficient_table(results, save = False):
    '''
    Returns a dataframe of coefficients, intercepts, and R squared values from linear regression 

    Arguments:
        results: results object from statsmodel.api.OLS function 
        save: boolean, whether or not to save dataframe as a csv file
            default = False, do not save 

    Returns:
        coefficients: pd.DataFrame, dataframe of coefficients, intercepts and R squared values
    '''

    # create a dataframe to house the coefficients 
    ols_index = results['NGCC'].params.index.to_list()
    ols_index.remove('const')
    ols_index.append('Intercept')
    ols_index.append('Rsq')
    ols_coef_table = pd.DataFrame(index = ols_index, columns = ['NGCC', 'SOFC', 'NGCC+SOEC', 'rSOC', 'SOFC+SOEC', 'SOEC'])

    # create a correspondinh table of p values to determine statistical significance of coefficients
    ols_p_table = pd.DataFrame(index = ols_index[:-1], columns = ['NGCC', 'SOFC', 'NGCC+SOEC', 'rSOC', 'SOFC+SOEC', 'SOEC'])

    for key in results.keys():
        ols_coef_table[key] = np.append(results[key].params.to_list()[1:], [results[key].params.to_list()[0], results[key].rsquared])
        ols_p_table[key] = np.append(results[key].pvalues.to_list()[1:], [results[key].pvalues.to_list()[0]])

    # append values with * if they are statistically significant
    for i in ols_coef_table.index[:-1]:
        for c in ols_coef_table.columns:
                # check if results are statistically significant
                if ols_p_table.at[i,c] <= 0.05:
                    ols_coef_table.at[i,c] = str(round(ols_coef_table.at[i,c],2)) + '*'
                else:
                    ols_coef_table.at[i,c] = str(round(ols_coef_table.at[i,c],2))

    for c in ols_coef_table.columns:
        ols_coef_table.at['Rsq', c] = str(round(ols_coef_table.at['Rsq',c],2))     

    if save:
        ols_coef_table.to_csv('ols_coefficient_table.csv')

    return ols_coef_table

def plot_featurecorrelation(fdest_data = 'formatted_raw_data.csv', save = False):
    '''
    Creates a dataframe of feature correlations and plots it. 

    Arguments:
        fdest_data: string, directory of data file 
            default = 'formatted_raw_data.csv', data file in this directory
        save: boolean, whether or not to save the plot image and dataframe as a csv 
            default = False, do not save the plot or csv file

    Returns:
        corr_df: pd.DataFrame, dataframe of feature correlation values 
    '''

    # read data to pd.DataFrame 
    formatted_data = read_raw_data(fdest_data)

    features, all_profits = separate_profits_features(formatted_data)

    # standardize the features 
    standardize_variables(features)

    # calculate the correlations
    corr_df = features.corr()

    # remove the upper triangle 
    corr_np = corr_df.to_numpy(dtype = float)
    corr_np[np.triu_indices_from(corr_np, k=1)] = np.nan 

    # convert back to a dataframe 
    corr_df = pd.DataFrame(corr_np)

    # return a list of labels 
    feature_labels = features.columns

    # make figure
    fig, ax = plt.subplots(figsize = (10,10))

    # plot feature correlation
    plt.imshow(corr_df, cmap = 'GnBu', vmin = -1, vmax = 1)
    plt.xticks(range(0,len(feature_labels)), labels = feature_labels, rotation = 45, ha = 'right',fontsize = 14, fontweight = 'bold')
    plt.yticks(range(0,len(feature_labels)), labels = feature_labels, fontsize = 14, fontweight = 'bold')

    # plot title
    plt.title('Feature Correlation Matrix', fontsize = 25, fontweight = 'bold')

    # Tick labels
    for i in range(0,len(feature_labels)):
        for j in range(0,len(feature_labels)):
            if not np.isnan(corr_df.at[i,j]):
                
                text = ax.text(j, i, round(corr_df.at[i, j],2),
                            ha="center", va="center", color="k", fontsize = 10, fontweight = 'bold')


    if save:
        # save plot image 
        plt.savefig('feature_correlation_plot.png', dpi = 300, bbox_inches = 'tight')
        plt.savefig('feature_correlation_plot.pdf', dpi = 300, bbox_inches = 'tight')

        # save correlation data as csv 
        corr_df.to_csv('feature_correlation.csv')

    plt.show()

    return corr_df

def plot_parityplots(results, fdest_data = 'formatted_raw_data.csv', save = False):
    '''
    Plots the parity between the optimization results and the linear regression model

    Arguments:
        results: dict, dictionary of ols regression results objects 
        fdest_data: string, destination of data file 
            default = 'formatted_raw_data.csv', data file in this directory
        save: boolean, whether or not to save the plot 
            default = False, do not save the plot 

    Returns:
        None 
    '''

    # save formatted raw data 
    formatted_data = read_raw_data(fdest_data)

    # separate features from profits 
    features, profits = separate_profits_features(formatted_data)

    standardize_variables(features)
    # standardize profits 
    cases = standardize_profits(profits)

    # formatted label and color dictionaries
    colors = {'NGCC': 'r', 'SOFC' : 'b', 'NGCC+SOEC' : 'g', 'rSOC' : 'm', 'SOFC+SOEC' : 'darkorange', 'SOEC' : 'deeppink'}
    labels = {'NGCC': 'NGCC', 'SOFC' : 'SOFC', 'NGCC+SOEC' : 'NGCC + SOEC', 'rSOC' : 'rSOC', 'SOFC+SOEC' : 'SOFC + SOEC', 'SOEC' : 'SOEC'}

    # make a 6 panel figure and six ax objects 
    fig, axs = plt.subplots(nrows = 2, ncols = 3, sharex = True, sharey = True, figsize = (11,9))

    # create dict of axes indexes associated with each system concept
    axes_list = {'NGCC': axs[0,0], 'NGCC+SOEC': axs[1,0], 'SOFC': axs[0,1], 'rSOC': axs[1,1], 'SOEC': axs[0,2], 'SOFC+SOEC': axs[1,2]}

    # adjust subplot spacing
    plt.subplots_adjust(top = 0.95, bottom = 0.06, left = 0.09)

    # add axes labels 
    fig.supxlabel('Annual Profit from Rigorous Optimization', y = -0.001, fontsize = 20, fontweight = 'bold')
    fig.supylabel('Annual Profit Predicted by Linear Regression', x = -0.001, fontsize = 20, fontweight = 'bold')

    # loop through the results objects 
    for key in results.keys():

        # return predicted profit values 
        features_ = sm.add_constant(features)
        y_pred = results[key].predict(features_)

        # save profit as np array 
        y = cases[key].to_numpy()

        # plot the parity plot 
        # add identity line
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        axes_list[key].plot([min_val, max_val], [min_val, max_val], 'k-', linewidth = 2, zorder = 2)

        # plot scatter of data 
        axes_list[key].scatter(y, y_pred, color = colors[key], zorder = 1)

        # add R-squared value to plot
        r_squared = results[key].rsquared
        axes_list[key].text(0.05, 0.82, f'R$^{2}$ = {r_squared:.2f}', transform=axes_list[key].transAxes, fontsize=12, fontweight = 'bold', bbox = dict(facecolor = 'white', edgecolor = 'k'))

        # generate title based on the case number 
        axes_list[key].set_title((labels[key]), fontsize = 12, fontweight = 'bold')

        # x and y tick labels
        for l in (axes_list[key].get_xticklabels() + axes_list[key].get_yticklabels()):
            l.set_weight('bold')
            l.set_fontsize(16)

        # grid
        axes_list[key].grid(True)

    if save:
        plt.savefig('ols_parity.png', dpi = 300, bbox_inches = 'tight')
        plt.savefig('ols_parity.pdf', dpi = 300, bbox_inches = 'tight')

    plt.show()

    return 






