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
import importlib
importlib.reload(modular_MPCA_online)

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
pathToOutput="C://project-data/my-data-sets/SPE_trajectories/"

alignedTransformed = pd.read_pickle(os.path.join(pathToDataSets,'alignedBatches_reshaped.pkl'))
alignedFoamingSignal = pd.read_csv(os.path.join(pathToDataSets,'alignedFoamingSignal_scaled.csv'))

#Show plots, 1 = yes, 0 = no
plots_on = 1
#%%
#alignedTransformed = alignedTransformed.iloc[0:30,:]

#%%
allOutliers, badBatchesFoaming = MPCA_functions.abnormalBatchesIndex(alignedTransformed, alignedFoamingSignal)
sizeTest = 0.3
test_set_index = MPCA_functions.det_test_index(alignedTransformed, allOutliers, sizeTest)
#varArray = ['E20161', 'FC20161', 'FQ20107A', 'FQ20113', 'FQ20141', 'P20160',
#       'P20163', 'PC20162', 'PC20162_Y', 'T20163', 'TC20171A', 'F20107A',
#       'TZ20171A', 'TZ20171B', 'TZ20172B', 'TZ20173', 'F20141',
#       'FC20101_W', 'FC20102', 'FC20103A', 'FC20104', 'FC20105',
#       'FC20144_Y']
varArray = ['PC20162', 'FC20144', 'P20160', 'TC20171A', 'TC20172A', 'FC20102',
            'E20161']#, 'FC20161'] 

#varArray = ['E20161', 'FQ20113', 'FQ20141', 'P20160', 'PC20162', 'PC20162_Y',
#            'T20163', 'TC20171A', 'TZ20171A', 'TZ20171B', 'TZ20172B', 'F20107A',
#            'TZ20173', 'F20141', 'FC20101_W', 'FC20102', 'FC20104', 'FC20105',
#            'FC20144_Y', 'FQ20107A']
            
#varArray = ['E20161', 'FC20103A', 'FC20104', 'FC20105', 'FC20144', 'FC20144_W',
#       'FC20144_Y', 'FC20160', 'FC20161', 'FC20161_W', 'FQ20107A',
#       'F20107A', 'FQ20107C', 'FQ20113', 'FQ20141', 'P20160', 'P20163',
#       'PC20162', 'PC20162_W', 'PC20162_Y', 'S20161', 'T20161', 'F20107B',
#       'T20163', 'TC20171A', 'TC20172A', 'TC20172A_W', 'TDZ20175A',
#       'TDZ20175B', 'TZ20171A', 'TZ20171B', 'TZ20172A', 'TZ20172B',
#       'F20107C', 'TZ20173', 'F20113', 'F20141', 'F20165', 'FC20101_W',
#       'FC20102']

# no_components determined via cross-validation
no_components = 10
alphaQ = 0.99
alphaD = 0.9
alpha_online_Q = 0.9999
alpha_online_D = 0.99
#%%
def main():
    # Split in test & training set and learn PCA model    
    batchesNormal, batchesTest, batchesOutlier, columnsValues = split_data(
        varArray, alignedTransformed, allOutliers, badBatchesFoaming, test_set_index)
    
    batchesNormalScaled, batchesTestScaled, batchesOutlierScaled, mean, std = standardize_data(
        batchesNormal, batchesTest, batchesOutlier, columnsValues)

    principal_components, eigvalues_full_model = build_PCA_model(
        batchesNormalScaled, no_components, columnsValues)

    scores, scoresTest, scoresBadBatches = calculate_scores(
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
#     confusion_matrix_Q = MPCA_functions.buildConfusionMatrix(
#         classification_Q_bad_batches, classification_Q_test)
#==============================================================================
      
    # Look at Hotelling statistic
    covMatrix = determine_covariance_matrix(scores)
    
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
#     confusion_matrix_D = MPCA_functions.buildConfusionMatrix(
#         classification_D_bad_batches, classification_D_test)
#     
#     print('\n' + 'Confusion Matrix Q' + '\n')    
#     print(confusion_matrix_Q) 
#     print('\n' + 'Confusion Matrix D' + '\n')    
#     print(confusion_matrix_D)   
#==============================================================================
    
    SPE_bad_batches, SPEVar_bad_batches, scores_online_Bad = modular_MPCA_online.calculate_online_SPE(
        batchesOutlier, principal_components, columnsValues, mean, std)
    SPE_normal_batches, SPEVar_normal_batches, scores_online_Normal = modular_MPCA_online.calculate_online_SPE(
        batchesNormal, principal_components, columnsValues, mean, std)
    SPE_test_batches, SPEVar_test_batches, scores_online_Test = modular_MPCA_online.calculate_online_SPE(
        batchesTest, principal_components, columnsValues, mean, std)
        
    SPE_control_limit_online = modular_MPCA_online.calculate_online_control_SPE(SPE_normal_batches, alpha_online_Q)
    SPE_confidence_matrix, SPE_rates = modular_MPCA_online.det_online_classification_SPE(
        SPE_normal_batches, SPE_bad_batches, SPE_test_batches, SPE_control_limit_online, alignedFoamingSignal)
    
    D_confidence_matrix, D_rates = modular_MPCA_online.det_online_classification_D(scores_online_Normal, 
                                                                                   scores_online_Bad, 
                                                                                   scores_online_Test, 
                                                                                   principal_components, 
                                                                                   covMatrix, 
                                                                                   alpha_online_D, 
                                                                                   alignedFoamingSignal)
    
    print(SPE_confidence_matrix,SPE_rates,D_confidence_matrix, D_rates)
    
#%%    
def split_data (var, data, outlier, outlier_foaming, test_index):
    data = data.loc[:,idx[:,var]]
    columnsValues = pd.MultiIndex.from_tuples(data.columns.values, names=['time', 'variables'])
    
    dataNormalAll = data[np.logical_not(data.index.isin(outlier))] 
    
    dataNormal = dataNormalAll[np.logical_not(dataNormalAll.index.isin(test_index))]
    dataNormal.columns = columnsValues
    dataTest = dataNormalAll[(dataNormalAll.index.isin(test_index))]
    dataTest.columns = columnsValues
    dataOutlier = data[data.index.isin(outlier_foaming)]
    dataOutlier.columns = columnsValues
    
    return(dataNormal, dataTest, dataOutlier, columnsValues)

#%%
def standardize_data (data, dataTest, dataOutlier, columns):
    mean = data.mean(axis = 0)
    std = data.std(axis = 0)
    # Replace all elements in standard deviation equal to zero with 1, for later use
    std[std == 0] = 1e-15
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
    
    scores = MPCA_functions.calculateScores(dataScaled, principal_components, 
                                            principal_components.index)
    
    scoresOutliers = MPCA_functions.calculateScores(dataOutlierScaled, principal_components,
                                                    principal_components.index)                           
    scoresTest = MPCA_functions.calculateScores(dataTestScaled, principal_components,
                                                principal_components.index)                           
    scoresBadBatches = scoresOutliers[scoresOutliers.index.isin(outlier)]
    return (scores, scoresTest, scoresBadBatches)

#%%
def calculate_Q_statistic (scores, principal_components, data): 
    ResidualMatrixSquared = (MPCA_functions.calcErrorMat(scores, principal_components, data, data.columns.values)**2)
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

if __name__ == '__main__':
    main()

 
 

          