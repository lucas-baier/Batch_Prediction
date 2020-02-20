# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:07:48 2017

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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os.path
import scipy.stats as sp_stats
import MPCA_functions
import random

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
scenarioSteam = [90,105,107,133,165,166,171,185,193,240]
scenarioPostSteam = [14,27,40,67,92,100,102,103,110,130,132]
scenarioOutlier = [26,244] 

pathToDataAligned="C:/project-data/my-data-sets/aligned"
pathToSPE="C://project-data/my-data-sets/SPE_trajectories/"

#Show plots, 1 = yes, 0 = no
plots_on = 1

#%%

def calculateSPE2 (scores, pComp, data, i, nVar):
    projection = np.dot(scores, pComp)
    SPEVar = ((data[(i*nVar):((i+1)*nVar)] - projection[(i*nVar):((i+1)*nVar)])**2)
    SPE = SPEVar.sum()
    return (SPE, SPEVar)
    
#%%
def calculate_online_SPE(data, principal_components, columnsValues, mean, std):
    noVar = len(data.columns.levels[1])
    scoresColumns = pd.MultiIndex.from_product([data.columns.levels[0],principal_components.index], names = ['time', 'components'])
        
    SPE_df = pd.DataFrame()
    scoresBatches_df = pd.DataFrame(columns = scoresColumns)
    SPEVar_df = pd.DataFrame(columns = columnsValues)
    
    for batch in data.index:
        batchData = data.loc[batch]
        batchScaled = MPCA_functions.standardizeInput(batchData, mean, std, columnsValues)
        resultSPE = {}  
        listSPE = []
        listScores = []
        for k in range(len(data.columns.levels[0])):
            if k < (len(data.columns.levels[0])-1):
                batchScaledDev = MPCA_functions.copyDevVector(batchScaled, k)
                scoresBatch = np.dot(batchScaledDev, principal_components.T)            
                SPE, SPEVar = calculateSPE2(scoresBatch, principal_components, batchScaledDev, k, noVar)
                resultSPE[k] = SPE
            else:
                resultSPE[k] = 0
                scoresBatch = np.dot(batchScaled, principal_components.T)
            listSPE = listSPE + SPEVar.tolist()
            listScores = listScores + scoresBatch.tolist()
    
        badBatchesSPEVarSeries = pd.Series(listSPE, index = columnsValues, name = batch)
        scoresBatchesSeries = pd.Series(listScores, index = scoresColumns, name = batch)
        trajectorySPE = pd.Series(resultSPE, name = batch)
    
        SPE_df = SPE_df.append(trajectorySPE)
        SPEVar_df = SPEVar_df.append(badBatchesSPEVarSeries)
        scoresBatches_df = scoresBatches_df.append(scoresBatchesSeries)
        
    return (SPE_df, SPEVar_df, scoresBatches_df)

#%%
def calculate_online_control_SPE (normalBatchesSPE, alpha):

#    def calculatePar (x):
#        return (x.mean(), x.var())
        
    meanNormal = {}
    varianceNormal = {}
    for i in normalBatchesSPE.columns.values:
        if (i == 0):
            x = pd.concat([normalBatchesSPE.iloc[:,i], normalBatchesSPE.iloc[:,i+1]])
        if (i > 0 and i < len(normalBatchesSPE.columns.values)-1):
            x = pd.concat([normalBatchesSPE.iloc[:,i-1], normalBatchesSPE.iloc[:,i], normalBatchesSPE.iloc[:,i+1]])
        if (i == len(normalBatchesSPE.columns.values)-1):
            x = pd.concat([normalBatchesSPE.iloc[:,i-1], normalBatchesSPE.iloc[:,i]])
        mean, var = (x.mean(), x.var())
        meanNormal[i] = mean
        varianceNormal[i] = var
        
    meanNormal = pd.DataFrame.from_records([meanNormal], index = ['mean'])
    varianceNormal = pd.DataFrame.from_records([varianceNormal], index = ['variance'])
    
    g = pd.DataFrame(((varianceNormal.loc['variance',:])/(2* meanNormal.loc['mean',:])), columns = ['g']).T
    h = pd.DataFrame(((2*(meanNormal.loc['mean',:]**2))/(varianceNormal.loc['variance',:])), columns = ['h']).T
    
    parNormal = pd.concat([meanNormal,varianceNormal, g, h])
    
    SPEAlpha = pd.Series((parNormal.loc['variance',:]/(2*parNormal.loc['mean',:]))*sp_stats.chi2.ppf(
        q = alpha, df = ((2*parNormal.loc['mean',:]**2)/(parNormal.loc['variance',:]))), 
        name = 'SPE_Control_Limit_{}'.format(alpha) )

    return(SPEAlpha)

#%%

def det_online_classification_SPE (normalBatchesSPE, badBatchesSPE, testBatchesSPE, SPEAlpha, alignedFoamingSignal):
    classificationNormalBatches = MPCA_functions.det_classification(normalBatchesSPE, SPEAlpha)    
    classificationBadBatches = MPCA_functions.det_classification(badBatchesSPE, SPEAlpha, alignedFoamingSignal)
    classificationTestBatches = MPCA_functions.det_classification(testBatchesSPE, SPEAlpha)
    
    confidence_matrix = MPCA_functions.buildConfusionMatrix(classificationBadBatches, classificationTestBatches)
    rates = MPCA_functions.calcRates(classificationBadBatches, classificationTestBatches)
    
    return(confidence_matrix, rates)
 
#%%

def calculateDOnline(scores, I, R, invMatrix):    
    result = ((np.dot(np.dot(scores.values, invMatrix),(scores.T).values))*I*(I-R))/(R*((I**2)-1))
    return (result)


def calculate_online_control_D(data_normal, principal_components, alpha): 
    I = len(data_normal.index)
    R = len(principal_components.index)
    control_limit = sp_stats.f.ppf(alpha, (R), (I-R))
    return(control_limit)
  

#%%  
def calc_classification_D (scores_batches, covariance_matrix, control_value, scores_normal, principal_components, aligned_foaming_signal = None):
    I = len(scores_normal.index)
    R = len(principal_components.index)
    inverseMatrix = np.linalg.inv(covariance_matrix)
    classification = {}  
    for batch in scores_batches.index: 
        values = scores_batches.loc[batch].unstack()
        values = values[list(principal_components.index.values)]
        Dvalues = values.apply(calculateDOnline, axis = 1, args = (I,R,inverseMatrix)) 
        if (aligned_foaming_signal is not None):
            foamP = MPCA_functions.foamingPoint(aligned_foaming_signal, batch)
            Dvalues = Dvalues[0:foamP-1]        
        binTable = Dvalues > control_value
        sumFailures = binTable.sum()
        if (sumFailures > 0):
            boolean = True
        else:
            boolean = False 
        classification[batch] = boolean
    
    predFaults = sum(classification.values())
    predNoFaults = len(classification) - predFaults
        
    batches_classification = (predFaults, predNoFaults)
    return (batches_classification)

#%%

def det_online_classification_D(scores_normal, scores_bad, scores_test, principal_components, cov_matrix, alpha, aligned_foaming_signal ):
    D_control_limit_online = calculate_online_control_D(
        scores_normal, principal_components, alpha)    
    classification_online_bad = calc_classification_D(scores_bad, 
                                                      cov_matrix, D_control_limit_online,
                                                      scores_normal, principal_components,
                                                      aligned_foaming_signal)
    classification_online_test = calc_classification_D(scores_test, 
                                                      cov_matrix, D_control_limit_online,
                                                      scores_normal, principal_components)
    
    confidence_matrix = MPCA_functions.buildConfusionMatrix(classification_online_bad, classification_online_test)
    rates = MPCA_functions.calcRates(classification_online_bad, classification_online_test)
    return(confidence_matrix, rates)

#%%
#==============================================================================
# classification = {}
# 
# for batch in scoresBatchesTest.index: 
#     values = (scoresBatchesTest.loc[idx[batch], idx[:,:]]).unstack()
#     Dvalues = values.apply(calculateDOnline, axis = 1, args = (I,R,inverseMatrix))  
#     binTable = Dvalues > controlValue
#     sumFailures = binTable.sum()
#     if (sumFailures > 0):
#         boolean = True
#     else:
#         boolean = False 
#     classification[batch] = boolean
#     predFaults = sum(classification.values())
#     predNoFaults = len(classification) - predFaults
#     
# testBatchesDClassification = (predFaults, predNoFaults)
# #%%
# confMatrixD = MPCA_functions.buildConfusionMatrix(badBatchesDClassification, testBatchesDClassification)
# kpiD = MPCA_functions.calcRates(badBatchesDClassification, testBatchesDClassification)
# confMatrixD
# kpiD
# 
#==============================================================================
