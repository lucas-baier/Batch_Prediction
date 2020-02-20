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
pathToDataSets="C:/project-data/my-data-sets/BASF_data_new_processed/"
pathToOutput="C:/project-data/my-data-sets/var_test_new_data/"

alignedTransformed = pd.read_pickle(os.path.join(pathToDataSets,'alignedBatches_reshaped_var_all.pkl'))
alignedFoamingSignal = pd.read_csv(os.path.join(pathToDataSets,'alignedFoamingSignal_scaled_var_all.csv'))

#Show plots, 1 = yes, 0 = no
plots_on = 1

#%%
allOutliers, badBatchesFoaming = MCPA_functions.abnormalBatchesIndex(alignedTransformed, alignedFoamingSignal)
variablesAll =  ['E20161', 'FC20144','F20107A', 'F20113',
               'F20141', 'F20165', 'FC20102',
               'FC20103A', 'FC20104', 'FC20105', 'FC20144_Y',
               'FC20160', 'FC20161', 'P20160', 'P20163',
               'PC20162', 'PC20162_Y', 'S20161',
               'T20161', 'T20163', 'TC20171A', 'TC20172A', 'TZ20171A',
               'TZ20171B', 'TZ20172A', 'TZ20173']
      
#%%
no_components = 10
alphaQ = 0.99
alphaD = 0.9
alpha_online_Q = 0.9999
alpha_online_D = 0.99

test_set_index = MCPA_functions.test_index_old_batches
test_index_new = MCPA_functions.test_index_new_batches
test_set_index = test_set_index + test_index_new


#%%
def main(variablesAll):
        
    results = pd.DataFrame()   
    column_var_names = ['var %i' %i for i in range(1,len(variablesAll)+1)]
    var_basic = ['E20161', 'FC20144']
    
    for s in range(24):
        variablesAll = sorted(list(set(variablesAll) - set(var_basic)))
        for i in variablesAll:
            varArray = var_basic + [i]         
            
            index_names = column_var_names[0:len(varArray)]
            
            # Split in test & training set and learn PCA model                            
            batchesNormal, batchesTest, batchesOutlier, columnsValues = modular_MCPA_main.split_data(
                varArray, alignedTransformed, allOutliers, badBatchesFoaming, test_set_index)
            
            batchesNormalScaled, batchesTestScaled, batchesOutlierScaled, mean, std = standardize_data(
                batchesNormal, batchesTest, batchesOutlier, columnsValues)
        
            principal_components, eigvalues_full_model = build_PCA_model(
                batchesNormalScaled, no_components, columnsValues)
        
            scores, scoresTest, scoresBadBatches = calculate_scores(
                batchesNormalScaled, principal_components, batchesOutlierScaled, 
                batchesTestScaled, columnsValues, badBatchesFoaming)
    
              
            # Look at Hotelling statistic
            covMatrix = determine_covariance_matrix(scores)
            
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
            results_try['Accuracy_Q'] = calc_output_accuracy(SPE_confidence_matrix)           
            results = results.append(results_try, ignore_index = True)   

        idx_max = results.iloc[-len(variablesAll):,:]['Accuracy_Q'].idxmax()
        i = (results[results.index == idx_max].iloc[0,-1])
        var_basic = var_basic + [i]

    results.to_pickle(os.path.join(pathToOutput, 'results_var_tests.pkl'))        
    print(results)
    
#%%    
def split_data (var, data, outlier, sizeTest):
    data = data.loc[:,idx[:,var]]
    columnsValues = pd.MultiIndex.from_tuples(data.columns.values, names=['time', 'variables'])

    dataNormalAll = data[np.logical_not(alignedTransformed.index.isin(outlier))]
    sizeTestSet = int(round(sizeTest * len(dataNormalAll.index),0))
    testIndex = random.sample(list(dataNormalAll.index.values), sizeTestSet)

    dataNormal = dataNormalAll[np.logical_not(dataNormalAll.index.isin(testIndex))]
    dataNormal.columns = columnsValues
    dataTest = dataNormalAll[(dataNormalAll.index.isin(testIndex))]
    dataTest.columns = columnsValues
    dataOutlier = data[data.index.isin(badBatchesFoaming)]
    dataOutlier.columns = columnsValues
    
    return(dataNormal, dataTest, dataOutlier, columnsValues)

#%%
def standardize_data (data, dataTest, dataOutlier, columns):
    mean = data.mean(axis = 0)
    std = data.std(axis = 0)
    # Replace all elements in standard deviation equal to zero with 1, for later use
    std[std == 0] = 1
    dataScaled = pd.DataFrame(preprocessing.scale(data), 
                                            index = data.index, 
                                            columns = columns)
                                            
    dataOutlierScaled = (dataOutlier - mean)/(std)
    dataOutlierScaled.columns = columns  
    dataTestScaled = (dataTest - mean)/(std)
    dataTestScaled.columns = columns
    
    return(dataScaled, dataTestScaled, dataOutlierScaled, mean, std)
    
#%%    
def build_PCA_model (dataScaled, no_components, columns):
    #Build full model    
    allComponents = len(dataScaled.index)
    pca = decomposition.PCA(n_components = allComponents)
    pca.fit(dataScaled)
    eigenvaluesAll = pca.explained_variance_
    
    #no_components based on results of cross-valiadtion, reduced model
    pca = decomposition.PCA(n_components = no_components)
    pca.fit(dataScaled)
    np.sum(pca.explained_variance_ratio_)
    columnnamesPC = ["PC%i" %s for s in range(1,pca.n_components_ + 1)]
    prinComponents = pd.DataFrame(pca.components_, index = columnnamesPC, columns = columns)
    return (prinComponents, eigenvaluesAll)

#%%    
def calculate_scores (dataScaled, principal_components, dataOutlierScaled, dataTestScaled,
                      columns, outlier):
    
    scores = MCPA_functions.calculateScores(dataScaled, principal_components, 
                                            principal_components.index)
    
    scoresOutliers = MCPA_functions.calculateScores(dataOutlierScaled, principal_components,
                                                    principal_components.index)                           
    scoresTest = MCPA_functions.calculateScores(dataTestScaled, principal_components,
                                                principal_components.index)                           
    scoresBadBatches = scoresOutliers[scoresOutliers.index.isin(outlier)]
    return (scores, scoresTest, scoresBadBatches)

#%%
def calculate_Q_statistic (scores, principal_components, data): 
    ResidualMatrixSquared = (MCPA_functions.calcErrorMat(scores, principal_components, data, data.columns.values)**2)
    Q = pd.DataFrame(ResidualMatrixSquared.sum(1), columns = ['SumOfSquares'])
    return (Q)

#%%    
def control_limit_Q (eigenvalues, alpha, n_comp):
    theta = dict()
    for i in range(1,4):
        x = eigenvalues[n_comp:]
        theta[i] = np.power(x,i).sum()
     
    h0 = 1- ((2*theta[1]*theta[3])/(3*(np.power(theta[2],2))))
    cAlpha = sp.stats.norm.ppf(alpha)
    QAlpha = theta[1]*np.power((((h0*cAlpha*np.power(2*theta[2],0.5))/(theta[1]))+
                               1+((theta[2]*h0*(h0-1))/(np.power(theta[1],2)))),(1/h0))
    return(QAlpha)
 
#%%
def det_classification(data, control_limit):

    classification = {}
    for i in data.index.values:  
        binTable = (data[data.index == i].values > control_limit)
        sumFailures = binTable.sum()
        if (sumFailures > 0):
            boolean = True
        else:
            boolean = False
        classification[i] = boolean  
    predFaults = sum(classification.values())
    predNoFaults = len(classification) - predFaults
    return (predFaults, predNoFaults)

#%%
def determine_covariance_matrix(scores):
    covMatrix = np.cov(scores.T)
    covMatrix[covMatrix < 0.00001] = 0
    return (covMatrix)

#%%    
def calculate_hotelling(scores, covMatrix):
    invCovMatrix = np.linalg.inv(covMatrix)
    
    def calculateD(scores, I, invMatrix):    
        result = ((np.dot(np.dot(scores.values, invMatrix),(scores.T).values))*I)/((I-1)**2)
        return (result)
         
    DS = scores.apply(calculateD, args = (len(scores.index), invCovMatrix), axis = 1)
    return(DS)

#%%
def control_limit_D (scores, alpha, n_comp):
    I = len(scores.index)
    R = n_comp
    nominator = ((R/(I-R-1))*sp.stats.f.ppf(alpha, (R), (I-R-1)))
    denominator = (1+(R/(I-R-1))*sp.stats.f.ppf(alpha, (R), (I-R-1)))
    hotellingStat = nominator/denominator
    return(hotellingStat)


#%%

def calc_output_accuracy(SPE_confidence_matrix):
    SPE_values = pd.Series([SPE_confidence_matrix.iloc[0,0],SPE_confidence_matrix.iloc[0,1],
                            SPE_confidence_matrix.iloc[1,0],SPE_confidence_matrix.iloc[1,1]], 
                            index = ['TP_Q','FN_Q','FP_Q','TN_Q'])
    Accuracy = (SPE_values[0] + SPE_values[3]) / (SPE_values.sum())
    SPE_values['Accuracy'] = Accuracy
    return (Accuracy)


#%%


if __name__ == '__main__':
    main(variablesAll)

  
##%%
#results = pd.read_pickle(os.path.join(pathToOutput, 'results_var_tests.pkl'))    
##%%
#
#x = results[(results['TPR_Q'] >0.9)] #& (results['TPR_D'] > 0.5) & (results['TNR_Q'] > 0.5)].iloc[:,14:] 
#
#x
#
##%%
#x_list = list(x.values.flatten())
#
#x_cleaned = [i for i in x_list if str(i) != 'nan']
##%%
#
#from collections import Counter
#dictCount = Counter(x_cleaned)
#
#countVar = pd.DataFrame.from_records(dictCount, index = ['Count']).T.sort_values('Count', ascending = False)
##%%
#countVar.to_pickle(os.path.join(pathToOutput,'countVar.pkl'))
##%%
#countVar.iloc[0:10,:].index

#%%
results = pd.read_pickle(os.path.join(pathToOutput, 'results_var_tests.pkl'))   
#%%
results

#%%
results[results['Accuracy_Q'] == results['Accuracy_Q'].max()].iloc[:,15:25]