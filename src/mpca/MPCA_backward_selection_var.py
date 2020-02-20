# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:04:57 2017

@author: delubai
"""
#
# GENERAL SETUP
#---------------------------
#
# import necessary libraries
#

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
import modular_MPCA_online
import modular_MPCA_main
import itertools

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

PYTHONHASHSEED = 0
random.seed(0)

#
# PATH TO DATA - MODIFY IT APPROPRIATELY!!!
#
pathToDataSets="C:/project-data/my-data-sets/aligned"
pathToOutput="C://project-data/my-data-sets/var_test/"

alignedTransformed = pd.read_pickle(os.path.join(pathToDataSets,'alignedBatches_reshaped.pkl'))
alignedFoamingSignal = pd.read_csv(os.path.join(pathToDataSets,'alignedFoamingSignal_scaled.csv'))

#Show plots, 1 = yes, 0 = no
plots_on = 1
#%%
allOutliers, badBatchesFoaming = MCPA_functions.abnormalBatchesIndex(alignedTransformed, alignedFoamingSignal)
varArray =  ['E20161', 'F20103B', 'F20107A', 'F20107B', 'F20107C', 'F20113',
               'F20141', 'F20165', 'FC20101_W', 'FC20102',
               'FC20103A', 'FC20104', 'FC20105', 'FC20144', 'FC20144_W', 'FC20144_Y',
               'FC20160', 'FC20161', 'FC20161_W',
               'FQ20107A', 'FQ20107C', 'FQ20113', 'FQ20141', 'P20160', 'P20163',
               'PC20162', 'PC20162_W', 'PC20162_Y', 'S20161',
               'T20161', 'T20163', 'TC20171A', 'TC20172A',
               'TC20172A_W', 'TDZ20175A', 'TDZ20175B', 'TZ20171A',
               'TZ20171B', 'TZ20172A', 'TZ20172B', 'TZ20173']

#varArray = ['E20161', 'FC20103A', 'FC20104', 'FC20105', 'FC20144', 'FC20144_W',
#       'FC20144_Y', 'FC20160', 'FC20161', 'FC20161_W', 'FQ20107A',
#       'F20107A', 'FQ20107C', 'FQ20113', 'FQ20141', 'P20160', 'P20163',
#       'PC20162', 'PC20162_W', 'PC20162_Y', 'S20161', 'T20161', 'F20107B',
#       'T20163', 'TC20171A', 'TC20172A', 'TC20172A_W', 'TDZ20175A',
#       'TDZ20175B', 'TZ20171A', 'TZ20171B', 'TZ20172A', 'TZ20172B',
#       'F20107C', 'TZ20173', 'F20113', 'F20141', 'F20165', 'FC20101_W',
#       'FC20102']

#varArray = ['PC20162', 'FC20144', 'P20160', 'TC20171A', 'TC20172A', 'FC20102', 'E20161'] 
#==============================================================================
# sel = []
# noRepeat = 8
# for i in range(5,len(variablesAll)+1):
#     selLocal = []    
#     
#     for j in range(noRepeat):    
#         selection = random.sample(variablesAll, i)
#         selLocal.append(selection)
#     
#     sel.append(selLocal)
#==============================================================================

column_var_names = ['var %i' %i for i in range(1,len(varArray)+1)]

test_index = modular_MCPA_main.test_set_index
no_components = modular_MCPA_main.no_components
alphaQ = modular_MCPA_main.alphaQ
alphaD = modular_MCPA_main.alphaD
alpha_online_Q = modular_MCPA_main.alpha_online_Q
alpha_online_D = modular_MCPA_main.alpha_online_D


variable_remove = ''

#%%
def main(var):
    
    varArray = var    
    results = pd.DataFrame()   
    for i in range(38):                            
            
        index_names = column_var_names[0:len(varArray)]
        
        # Split in test & training set and learn PCA model                            
        batchesNormal, batchesTest, batchesOutlier, columnsValues = modular_MCPA_main.split_data(
            varArray, alignedTransformed, allOutliers, badBatchesFoaming, test_index)
        
        batchesNormalScaled, batchesTestScaled, batchesOutlierScaled, mean, std = modular_MCPA_main.standardize_data(
            batchesNormal, batchesTest, batchesOutlier, columnsValues)
    
        principal_components, eigvalues_full_model = modular_MCPA_main.build_PCA_model(
            batchesNormalScaled, no_components, columnsValues)
    
        scores, scoresTest, scoresBadBatches = modular_MCPA_main.calculate_scores(
            batchesNormalScaled, principal_components, batchesOutlierScaled, 
            batchesTestScaled, columnsValues, badBatchesFoaming)
       
        # Look at Q/SPE statistic    
    #==============================================================================
    #     Q_normal = calculate_Q_statistic(scores, principal_components, 
    #                                     batchesNormalScaled)
    #     Q_test = calculate_Q_statistic(scoresTest, principal_components, 
    #                                     batchesTestScaled)
    #     Q_bad_batches = calculate_Q_statistic(scoresBadBatches, principal_components, 
    #                                     batchesOutlierScaled)
    #                                     
    #     control_limit = control_limit_Q(eigvalues_full_model, alphaQ, no_components )
    #     
    #     classification_Q_normal = det_classification(Q_normal, control_limit) 
    #     classification_Q_test = det_classification(Q_test, control_limit)
    #     classification_Q_bad_batches = det_classification(Q_bad_batches, control_limit)
    #     confusion_matrix_Q = MCPA_functions.buildConfusionMatrix(
    #         classification_Q_bad_batches, classification_Q_test)
    #==============================================================================
          
        # Look at Hotelling statistic
        covMatrix = modular_MCPA_main.determine_covariance_matrix(scores)
        
    #==============================================================================
    #     D_normal = calculate_hotelling(scores, covMatrix)
    #     D_test = calculate_hotelling(scoresTest, covMatrix)
    #     D_bad_batches = calculate_hotelling(scoresBadBatches, covMatrix)
    #     
    #     d_control_limit = control_limit_D(scores, alphaD, no_components)
    #     
    #     classification_D_normal = det_classification(D_normal, d_control_limit)
    #     classification_D_test = det_classification(D_test, d_control_limit)
    #     classification_D_bad_batches = det_classification(D_bad_batches, d_control_limit)
    #     confusion_matrix_D = MCPA_functions.buildConfusionMatrix(
    #         classification_D_bad_batches, classification_D_test)
    #     
    #     print('\n' + 'Confusion Matrix Q' + '\n')    
    #     print(confusion_matrix_Q) 
    #     print('\n' + 'Confusion Matrix D' + '\n')    
    #     print(confusion_matrix_D)   
    #==============================================================================
        
        SPE_bad_batches, SPEVar_bad_batches, scores_online_Bad = modular_MCPA_online.calculate_online_SPE(
            batchesOutlier, principal_components, columnsValues, mean, std)
        SPE_normal_batches, SPEVar_normal_batches, scores_online_Normal = modular_MCPA_online.calculate_online_SPE(
            batchesNormal, principal_components, columnsValues, mean, std)
        SPE_test_batches, SPEVar_test_batches, scores_online_Test = modular_MCPA_online.calculate_online_SPE(
            batchesTest, principal_components, columnsValues, mean, std)
            
        SPE_control_limit_online = modular_MCPA_online.calculate_online_control_SPE(SPE_normal_batches, alpha_online_Q)
        SPE_confidence_matrix, SPE_rates = modular_MCPA_online.det_online_classification_SPE(
            SPE_normal_batches, SPE_bad_batches, SPE_test_batches, SPE_control_limit_online, alignedFoamingSignal)
        
        D_confidence_matrix, D_rates = modular_MCPA_online.det_online_classification_D(scores_online_Normal, 
                                                                                       scores_online_Bad, 
                                                                                       scores_online_Test, 
                                                                                       principal_components, 
                                                                                       covMatrix, 
                                                                                       alpha_online_D, 
                                                                                       alignedFoamingSignal)
        
        # prepare results for the output table
        SPE_values = pd.Series([SPE_confidence_matrix.iloc[0,0],SPE_confidence_matrix.iloc[0,1],
                                SPE_confidence_matrix.iloc[1,0],SPE_confidence_matrix.iloc[1,1]], 
                                index = ['TP_Q','FN_Q','FP_Q','TN_Q'])
        SPE_results = SPE_values.append(SPE_rates.rename({'TPR':'TPR_Q','TNR':'TNR_Q','Precision':'Precision_Q'}))              
                      
        D_values = pd.Series([D_confidence_matrix.iloc[0,0],D_confidence_matrix.iloc[0,1],
                              D_confidence_matrix.iloc[1,0],D_confidence_matrix.iloc[1,1]], 
                                index = ['TP_D','FN_D','FP_D','TN_D'])
        D_results = D_values.append(D_rates.rename({'TPR':'TPR_D','TNR':'TNR_D','Precision':'Precision_D'}))
                        
        results_numbers = SPE_results.append(D_results)
        var_series = pd.Series(varArray, index = index_names) 
        results_try = results_numbers.append(var_series)
        results = results.append(results_try, ignore_index = True)
        
        variable_remove = det_var_to_remove(principal_components)            
        varArray.remove(variable_remove)  
    
    results.to_pickle(os.path.join(pathToOutput, 'results_var_backward_selection.pkl'))        
    print(results)
    
#%%  

def det_var_to_remove(principal_comp):
    d = {}
    for i in varArray:
        d[i] = (principal_comp.loc[idx[:],idx[:,i]]**2).sum(axis = 1).sum()
    index = max (d, key = d.get)
    r = {}
    for key, values in d.items():
         r[key] = values / d[index]
             
    x = pd.DataFrame(r, index = ['VJ']).T.sort_values(by = 'VJ', ascending = False)
    variable_remove = x.iloc[-1].name
    return(variable_remove)
    
#%%

if __name__ == '__main__':
    main(varArray)
else:
	print ('')
 

#%%
results = pd.read_pickle(os.path.join(pathToOutput, 'results_var_backward_selection.pkl'))    
#%%

results['TP_Q'] + results['TP_Q']


#%% 
accuracy_Q = (results['TP_Q'] + results['TN_Q']) / (results['TP_Q'] + results['FP_Q'] + results['TN_Q'] + results['FN_Q'])
accuracy_Q

accuracy_D = (results['TP_D'] + results['TN_D']) / (results['TP_D'] + results['FP_D'] + results['TN_D'] + results['FN_D'])
accuracy_D

#%%

results['accuracy_Q'] = accuracy_Q
results['accuracy_D'] = accuracy_D

#%%
x = np.linspace(41,4,38)
fig, ax = plt.subplots()
ax.set_ylim(0.4,0.8)
ax.set_ylabel('Accuracy')
plt.plot(x, accuracy_Q, color = 'steelblue')
#plt.plot(x, accuracy_D, color = 'orange')
#%%

results.iloc[-10:,:].dropna(axis = 1, how = 'all').iloc[:,-15:]


#%%
#
#l1 = ['accuracy_Q','accuracy_D','TP_Q','FN_Q','FP_Q','TN_Q', 'TPR_Q','TNR_Q','Precision_Q']
#l2 = ['TP_D','FN_D','FP_D','TN_D', 'TPR_D', 'TNR_D', 'Precision_D']
#
#col = l1 + l2 + column_var_names
#results.columns = col
#%%
(results.iloc[1,:14])

#%%
results[['var 1', 'var 2', 'var 3', 'var 4', 'var 5']]
#%%
p = (results['TP_Q'] + results['TN_Q'])/52 
#%%

seriesVar = results.iloc[21:,:]
seriesVar
#%%
seriesVar = seriesVar[pd.notnull(seriesVar)].values
seriesVar
#%%

#x = np.linspace(41,4,38)
#fig, ax = plt.subplots()
#plt.plot(x, results['TPR_Q'])
#plt.plot(x, results['TPR_D'], color = 'orange')
#plt.plot(x, p, color = 'red')
#ax.set_xlim([15,20])
#%%
#seriesVar = (results[pd.notnull(results['var 17'])].iloc[18,14:])
#seriesVar = seriesVar[pd.notnull(seriesVar)].values
#seriesVar
##%%
#seriesVar = (results[pd.notnull(results['var 17'])].iloc[-1,14:])
#seriesVar = seriesVar[pd.notnull(seriesVar)].values
#seriesVar

# #%%
# x_list = list(x.values.flatten())
# 
# x_cleaned = [i for i in x_list if str(i) != 'nan']
# #%%
# 
# from collections import Counter
# dictCount = Counter(x_cleaned)
# 
# countVar = pd.DataFrame.from_records(dictCount, index = ['Count']).T.sort_values('Count', ascending = False)
# #%%
# countVar.to_pickle(os.path.join(pathToOutput,'countVar.pkl'))
# #%%
# countVar.iloc[0:10,:].index
#==============================================================================
#results.iloc[:,:14]
#results