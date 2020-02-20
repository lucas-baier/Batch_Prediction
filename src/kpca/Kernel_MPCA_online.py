# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:23:24 2017

@author: delubai
"""

import sys
sys.path.append('C:\project-data\my-codes\Spyder_workspace\MCPA_batch_data')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os.path
import random
from sklearn import preprocessing
from sklearn import decomposition
import MPCA_functions 
import modular_MPCA_main
import modular_MPCA_online
import KPCA_functions
import Kernel_MPCA
import importlib

#importlib.reload(Kernel_MPCA)

plt.rcParams['figure.figsize'] = (14, 10)

#
# print all output, not just the last expression
#
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# set the numbers of decimals to be displayed
pd.set_option('precision',3)

# use for simplified multiindex access
idx = pd.IndexSlice 

# Batches for the fault scenarios with foaming signal
scenarioSteam = [90,105,107,133,165,166,171,185,193,240,256,257,259,260,261,
                 268,284]
scenarioPostSteam = [14,27,40,67,92,100,102,103,110,130,132]
scenarioOutlier = [26,244] 

pathToDataSets="C://project-data/my-data-sets/data_new_processed/"
pathToOutput = "C:/project-data/my-data-sets/Kernel_MPCA_new/"

alignedTransformed = pd.read_pickle(os.path.join(pathToDataSets,'alignedBatches_reshaped.pkl'))
alignedFoamingSignal = pd.read_csv(os.path.join(pathToDataSets,'alignedFoamingSignal_scaled.csv'))

#Show plots, 1 = yes, 0 = no
plots_on = 1

#%%
varArray = [ 'E20161', 'FC20144', 'P20160','PC20162', 'TC20171A',
            'FC20102', 'TC20172A']

index_training = MPCA_functions.index_training_batches
index_test = MPCA_functions.index_test_batches
index_foaming = MPCA_functions.index_foaming_batches

no_components = 5

alphaQ = 0.99
alphaD = 0.99
alpha_online_Q = 0.9999
alpha_online_D = 0.99

degree_poly = 2
coef0_poly = 7.5

kernel_choice = 'rbf'
#%%
batchesNormal, batchesTest, batchesOutlier, columnsValues = Kernel_MPCA.split_data(
    varArray, alignedTransformed, index_training, index_test, index_foaming)

gamma_rbf = 1/(3.5*batchesNormal.shape[1])

batchesNormalScaled, batchesTestScaled, batchesOutlierScaled, mean, std = modular_MPCA_main.standardize_data(
    batchesNormal, batchesTest, batchesOutlier, columnsValues)

#%%
kernel_matrix_poly, kernel_centerer_poly = KPCA_functions.calculate_kernel_matrix_poly(batchesNormalScaled, degree_poly, coef0_poly)  
kernel_matrix_rbf, kernel_centerer_rbf = KPCA_functions.calculate_kernel_matrix_rbf(batchesNormalScaled, gamma_rbf)    

alphas_poly, lambdas_poly, alphas_full_poly, lambdas_full_poly = KPCA_functions.det_eigvectors(kernel_matrix_poly, no_components)
alphas_rbf, lambdas_rbf, alphas_full_rbf, lambdas_full_rbf = KPCA_functions.det_eigvectors(kernel_matrix_rbf, no_components)

scores_model_poly = KPCA_functions.calc_scores(kernel_matrix_poly, alphas_poly)
scores_model_rbf = KPCA_functions.calc_scores(kernel_matrix_rbf, alphas_rbf)
   
#%%

def calculate_SPE (scores, scores_full):
    SPE = (scores_full**2).sum(axis = 1) - (scores**2).sum(axis = 1)
    return (SPE.values[0])    
    
    
#%%
def calc_online_kernel_SPE (data, batches_normal_scaled, alphas, alphas_full, columns_values, kernel):
    columnnamesPC = ["PC%i" %(s) for s in range(1,alphas.shape[1] + 1)]
    scoresColumns = pd.MultiIndex.from_product([data.columns.levels[0],columnnamesPC], names = ['time', 'components'])
    
    SPE_df = pd.DataFrame()
    scores_df = pd.DataFrame(columns = scoresColumns)
    
    for batch in data.index:
        batchData = data.loc[batch]
        batchScaled = (MPCA_functions.standardizeInput(batchData, mean, std, columnsValues)
                        .rename(batch))
        list_SPE = []
        list_scores = []
        for k in range(len(data.columns.levels[0])):
            if k < (len(data.columns.levels[0])-1):
                batchScaledDev = pd.Series(MPCA_functions.copyDevVector(batchScaled, k), 
                                           index = columns_values, name = batch)
                if kernel == 'poly':
                    kernel_matrix = KPCA_functions.other_kernel_matrix_poly(batches_normal_scaled, 
                                                                            batchScaledDev, 
                                                                            kernel_centerer_poly, 
                                                                            degree_poly, 
                                                                            coef0_poly)
                if kernel == 'rbf':
                    kernel_matrix = KPCA_functions.other_kernel_matrix_rbf(batches_normal_scaled,
                                                                           batchScaledDev,
                                                                           kernel_centerer_rbf,
                                                                           gamma_rbf)                                                            
            else:
                batchScaled.name = batch            
                if kernel == 'poly':
                    kernel_matrix = KPCA_functions.other_kernel_matrix_poly(batches_normal_scaled, 
                                                                            batchScaled, 
                                                                            kernel_centerer_poly, 
                                                                            degree_poly, 
                                                                            coef0_poly)
                if kernel == 'rbf':
                    kernel_matrix = KPCA_functions.other_kernel_matrix_rbf(batches_normal_scaled,
                                                                           batchScaled,
                                                                           kernel_centerer_rbf,
                                                                           gamma_rbf)   
                                                                           
            scores = KPCA_functions.calc_scores(kernel_matrix, alphas)
            scores_full = KPCA_functions.calc_scores(kernel_matrix, alphas_full)
            SPE = calculate_SPE(scores, scores_full)
            
            list_scores = list_scores + pd.Series(scores.values[0]).tolist()
            list_SPE = list_SPE + [SPE]
            
        SPE_batch = pd.Series(list_SPE, name = batch)
        scores_batch = pd.Series(list_scores, index = scoresColumns, name = batch)
    
        SPE_df = SPE_df.append(SPE_batch)
        scores_df = scores_df.append(scores_batch)
        
    return (SPE_df, scores_df)

#%%

#==============================================================================
# SPE_foaming_batches, scores_foaming_batches = calc_online_kernel_SPE(batchesOutlier, 
#                                                                       batchesNormalScaled, 
#                                                                       alphas_poly, 
#                                                                       alphas_full_poly, 
#                                                                       columnsValues,
#                                                                       kernel_choice)
#                                                       
# SPE_normal_batches, scores_normal_batches = calc_online_kernel_SPE(batchesNormal,
#                                                                    batchesNormalScaled,
#                                                                    alphas_poly,
#                                                                    alphas_full_poly,
#                                                                    columnsValues,
#                                                                    kernel_choice)
#                                                                    
# SPE_test_batches, scores_test_batches = calc_online_kernel_SPE(batchesTest,
#                                                                batchesNormalScaled,
#                                                                alphas_poly,
#                                                                alphas_full_poly,
#                                                                columnsValues,
#                                                                kernel_choice)
#==============================================================================
#%%
                                                               
#==============================================================================
# SPE_normal_batches.to_pickle(os.path.join(pathToOutput,'SPE_normal_batches_7var_scaled.pkl'))
# SPE_test_batches.to_pickle(os.path.join(pathToOutput,'SPE_test_batches_7var_scaled.pkl'))
# SPE_foaming_batches.to_pickle(os.path.join(pathToOutput,'SPE_foaming_batches_7var_scaled.pkl'))
# 
# scores_normal_batches.to_pickle(os.path.join(pathToOutput,'scores_normal_batches_7var_scaled.pkl'))
# scores_test_batches.to_pickle(os.path.join(pathToOutput,'scores_test_batches_7var_scaled.pkl'))
# scores_foaming_batches.to_pickle(os.path.join(pathToOutput,'scores_foaming_batches_7var_scaled.pkl'))
#==============================================================================

#%%

SPE_normal_batches = pd.read_pickle(os.path.join(pathToOutput,'SPE_normal_batches_7var_scaled.pkl'))
SPE_test_batches = pd.read_pickle(os.path.join(pathToOutput,'SPE_test_batches_7var_scaled.pkl'))
SPE_foaming_batches = pd.read_pickle(os.path.join(pathToOutput,'SPE_foaming_batches_7var_scaled.pkl'))

scores_normal_batches = pd.read_pickle(os.path.join(pathToOutput,'scores_normal_batches_7var_scaled.pkl'))
scores_test_batches = pd.read_pickle(os.path.join(pathToOutput,'scores_test_batches_7var_scaled.pkl'))
scores_foaming_batches = pd.read_pickle(os.path.join(pathToOutput,'scores_foaming_batches_7var_scaled.pkl'))                                                            
#%%

confidence_limit_SPE = modular_MPCA_online.calculate_online_control_SPE(SPE_normal_batches, alpha_online_Q)                                                          
 #%%                                                           
def plot_online_kernel_SPE(batches_data, conf_limits_SPE, aligned_foaming_signal = None):                                                      

    for batch in batches_data.index:
        batch_data = batches_data[batches_data.index == batch].iloc[0,:]
        fig, ax1 = plt.subplots()
        _ = fig.suptitle('Batch {}'.format(batch))
        _ = ax1.plot(batch_data, label = 'SPE',marker = 'o', linestyle='-', markersize = 3)
        _ = ax1.plot(conf_limits_SPE, label = '{} confidence limit'.format(conf_limits_SPE.name))
        _ = ax1.set_xlabel('time (min)')
        _ = ax1.set_ylabel('SPE')
        #_ = ax1.set_ylim(0,500)
        #_ = ax1.set_xlim(210,235)
        if (aligned_foaming_signal is not None):                
            foamP = MPCA_functions.foamingPoint(aligned_foaming_signal, batch)        
            _ = ax1.axvline(foamP, color = 'red', label = 'First foaming signal')
        _ = plt.legend()
        _ = plt.show   

#%%
#plot_online_kernel_SPE(SPE_foaming_batches, confidence_limit_SPE, alignedFoamingSignal)
#plot_online_kernel_SPE(SPE_test_batches, confidence_limit_SPE)

#%%

def calc_online_kernel_D(scores, scores_model_poly, scores_model_rbf):
    if kernel_choice == 'poly':    
        covMatrix = np.cov(scores_model_poly.T)
    if kernel_choice == 'rbf':
        covMatrix = np.cov(scores_model_rbf.T)
    covMatrix[covMatrix < 0.00001] = 0    
    inverse_covariance_matrix = np.linalg.inv(covMatrix)
    df_D = pd.DataFrame()
    
    def calculate_D_stat (scores):
        result = np.dot(np.dot(scores, inverse_covariance_matrix), scores.T)
        return (result)    
    
    for batch in (scores.index):            
        scores_table = scores[scores.index == batch].T.unstack()    
        D_values = scores_table.apply(calculate_D_stat, axis = 1)
        D_values = pd.Series(D_values, name = batch)   
        df_D = df_D.append(D_values)        
    
    return (df_D)

#%%
def calc_online_kernel_D_control_limit(scores_normal, no_components, alpha_online_D):
    p = no_components
    I = len(scores_normal.index)    
    control_limit = (((p * (((I)**2)-1))/(I * (I - p))) * 
                    sp.stats.f.ppf(alpha_online_D, p, (I - p)))
    control_limit = pd.Series(control_limit, name = 'D confidence limit {}'.format(alpha_online_D))
    return (control_limit)

#%%               
def plot_online_kernel_D(D_values_table, control_limit, aligned_foaming_signal = None):
    
    for batch in (D_values_table.index):
        D_values = D_values_table[D_values_table.index == batch].iloc[0,:]        

        fig, ax1 = plt.subplots()
        _ = fig.suptitle('Batch %i' %batch)
        
        x = D_values.index
        y = D_values.values
        
        _ = ax1.plot(x, y, label = 'D',marker = 'o', linestyle='-', markersize = 3)
        _ = ax1.set_xlabel('time (min)')
        _ = ax1.set_ylabel('D')
        _ = ax1.set_ylim(0,3)
        #_ = ax1.set_xlim(300,360)
        _ = ax1.axhline(control_limit.values[0], color = 'black', label = 'F statistic')

        if (aligned_foaming_signal is not None):                
            foamP = MPCA_functions.foamingPoint(aligned_foaming_signal, batch)        
            _ = ax1.axvline(foamP, color = 'red', label = 'First foaming signal')
    
        _ = plt.legend()
        _ = plt.show  
#%%
D_foaming_batches = calc_online_kernel_D(scores_foaming_batches, scores_model_poly, scores_model_rbf)
D_normal_batches = calc_online_kernel_D(scores_normal_batches, scores_model_poly, scores_model_rbf)
D_test_batches = calc_online_kernel_D(scores_test_batches, scores_model_poly, scores_model_rbf)

#%%
control_limit_D = calc_online_kernel_D_control_limit(scores_normal_batches, no_components, alpha_online_D)
#%%   
plot_online_kernel_D(D_foaming_batches, control_limit_D, alignedFoamingSignal)
#plot_online_kernel_D(D_test_batches, control_limit_D)

#%%

SPE_class_foaming_batches = MPCA_functions.det_classification(SPE_foaming_batches, 
                                                              confidence_limit_SPE, 
                                                              alignedFoamingSignal)
SPE_class_test_batches = MPCA_functions.det_classification(SPE_test_batches, 
                                                           confidence_limit_SPE)
SPE_confusion_matrix = MPCA_functions.buildConfusionMatrix(SPE_class_foaming_batches, 
                                                           SPE_class_test_batches)
#%%

D_class_foaming_batches = MPCA_functions.det_classification(D_foaming_batches,
                                                            control_limit_D,
                                                            alignedFoamingSignal)
D_class_test_batches = MPCA_functions.det_classification(D_test_batches,
                                                         control_limit_D)
D_confusion_matrix = MPCA_functions.buildConfusionMatrix(D_class_foaming_batches,
                                                         D_class_test_batches)
                                                         
#%%
                                                         
def calc_output_accuracy(SPE_confidence_matrix):
    SPE_values = pd.Series([SPE_confidence_matrix.iloc[0,0],SPE_confidence_matrix.iloc[0,1],
                            SPE_confidence_matrix.iloc[1,0],SPE_confidence_matrix.iloc[1,1]], 
                            index = ['TP_Q','FN_Q','FP_Q','TN_Q'])
    Accuracy = (SPE_values[0] + SPE_values[3]) / (SPE_values.sum())
    SPE_values['Accuracy'] = Accuracy
    return (SPE_values)
    
#%%
    
calc_output_accuracy(SPE_confusion_matrix)