# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:52:15 2017

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
import sklearn as sk
import random
import MPCA_functions
import modular_MPCA_main
import modular_MPCA_online
import importlib
importlib.reload(MPCA_functions)

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

pathToDataSets="C:/project-data/my-data-sets/aligned"
pathToDataSets_new="C://project-data/my-data-sets/data_new_processed/"
pathToDataAligned="C:/project-data/my-data-sets/aligned"
pathToSPE="C://project-data/my-data-sets/SPE_trajectories_new/"

path_to_output = "C:/project-data/my-data-sets/Online_DTW_Data/"

alignedTransformed = pd.read_pickle(os.path.join(pathToDataSets_new,'alignedBatches_reshaped.pkl'))
alignedFoamingSignal = pd.read_csv(os.path.join(pathToDataSets_new,'alignedFoamingSignal_scaled.csv'))

#Show plots, 1 = yes, 0 = no
plots_on = 1

no_components = 10

# If scaling is not done with all training batches, it is possible to specify 
# the number of previous batches used for scaling

no_batches_for_scaling = 50
#%%  

varArray = ['E20161', 'FC20102', 'FC20144', 'P20160', 'PC20162', 'TC20171A',
            'TC20172A']#, 'FC20161']

alignedTransformed = alignedTransformed.loc[:,idx[:,varArray]]

allOutliers, badBatchesFoaming = MPCA_functions.abnormalBatchesIndex(alignedTransformed, alignedFoamingSignal)

test_set_index = MPCA_functions.index_test_batches

alignedTransformedNormal, alignedTransformedTest, alignedTransformedOutlier, columnsValues = modular_MPCA_main.split_data(
    varArray, alignedTransformed, allOutliers, badBatchesFoaming, test_set_index)

#%%
#alignedTransformedTest, alignedTransformedOutlier = MPCA_functions.det_test_foaming_batches_new(alignedTransformed[alignedTransformed.index > 247],
#                                                                                                        alignedFoamingSignal,
#                                                                                                        varArray)
##%%
#alignedTransformedNormal_new, alignedTransformedTest_new, alignedTransformedOutlier_new = MPCA_functions.det_normal_test_foaming_batches_new(alignedTransformed,
#                                                                                    alignedFoamingSignal,
#                                                                                    varArray, test_index_new)
#alignedTransformedNormal = pd.concat([alignedTransformedNormal, alignedTransformedNormal_new])
#alignedTransformedTest = pd.concat([alignedTransformedTest, alignedTransformedTest_new])
#alignedTransformedOutlier = pd.concat([alignedTransformedOutlier, alignedTransformedOutlier_new])

#%%
batchesNormalScaled, batchesTestScaled, batchesOutlierScaled, mean, std = modular_MPCA_main.standardize_data(
    alignedTransformedNormal, alignedTransformedTest, alignedTransformedOutlier, columnsValues)

#batchesNormalScaled, batchesTestScaled, batchesOutlierScaled = MPCA_functions.standardize_data_current(
#    alignedTransformedNormal, alignedTransformedTest, alignedTransformedOutlier, columnsValues, no_batches_for_scaling)
#    
principal_components, eigvalues_full_model = modular_MPCA_main.build_PCA_model(
    batchesNormalScaled, no_components, columnsValues)

scores, scoresTest, scoresBadBatches = modular_MPCA_main.calculate_scores(
    batchesNormalScaled, principal_components, batchesOutlierScaled, 
    batchesTestScaled, columnsValues, badBatchesFoaming)
  
covMatrix = modular_MPCA_main.determine_covariance_matrix(scores)
                                        
columnsValues = pd.MultiIndex.from_tuples(batchesNormalScaled.columns.values, names=['time', 'variables'])

aligned_transformed_for_scaling = pd.concat([alignedTransformedNormal, alignedTransformedTest]).sort_index()

#%%
cov_matrix = pd.DataFrame(covMatrix)
#==============================================================================
# principal_components.to_pickle(os.path.join(path_to_output, 'principal_components.pkl'))
# mean.to_pickle(os.path.join(path_to_output, 'mean.pkl'))
# std.to_pickle(os.path.join(path_to_output, 'std.pkl'))
# cov_matrix.to_pickle(os.path.join(path_to_output, 'covariance_matrix_scores.pkl'))
#==============================================================================

#%%

#==============================================================================
# import time
# start_time = time.time()
# 
# bad_batches_SPE, bad_batches_SPEVar, bad_batches_scores = modular_MPCA_online.calculate_online_SPE(
#     alignedTransformedOutlier, principal_components, columnsValues, mean, std)
# normal_batches_SPE, normal_batches_SPEVar, normal_batches_scores = modular_MPCA_online.calculate_online_SPE(
#     alignedTransformedNormal, principal_components, columnsValues, mean, std)
# test_batches_SPE, test_batches_SPEVar, test_batches_scores = modular_MPCA_online.calculate_online_SPE(
#     alignedTransformedTest, principal_components, columnsValues, mean, std)
# 
# #bad_batches_SPE, bad_batches_SPEVar, bad_batches_scores = MPCA_functions.calculate_online_SPE(
# #    alignedTransformedOutlier, principal_components, columnsValues, aligned_transformed_for_scaling, no_batches_for_scaling)
# #normal_batches_SPE, normal_batches_SPEVar, normal_batches_scores = MPCA_functions.calculate_online_SPE(
# #    alignedTransformedNormal, principal_components, columnsValues, aligned_transformed_for_scaling, no_batches_for_scaling)
# #test_batches_SPE, test_batches_SPEVar, test_batches_scores = MPCA_functions.calculate_online_SPE(
# #    alignedTransformedTest, principal_components, columnsValues,  aligned_transformed_for_scaling, no_batches_for_scaling)
# 
# print('Dauer %s' % (time.time() - start_time))
#==============================================================================


#%%

#==============================================================================
# normal_batches_SPE.to_pickle(os.path.join(pathToSPE,'SPE_NormalBatches_7var_scaled.pkl'))
# bad_batches_SPE.to_pickle(os.path.join(pathToSPE,'SPE_FoamingBatches_7var_scaled.pkl'))
# test_batches_SPE.to_pickle(os.path.join(pathToSPE,'SPE_TestBatches_7var_scaled.pkl'))
# normal_batches_scores.to_pickle(os.path.join(pathToSPE,'Scores_NormalBatches_7var_scaled.pkl'))
# bad_batches_scores.to_pickle(os.path.join(pathToSPE,'Scores_FoamingBatches_7var_scaled.pkl'))
# test_batches_scores.to_pickle(os.path.join(pathToSPE,'Scores_TestBatches_7var_scaled.pkl'))
# bad_batches_SPEVar.to_pickle(os.path.join(pathToSPE,'SPEVar_FoamingBatches_7var_scaled.pkl'))
# test_batches_SPEVar.to_pickle(os.path.join(pathToSPE,'SPEVar_TestBatches_7var_scaled.pkl'))
#==============================================================================

#%%
normal_batches_SPE = pd.read_pickle(os.path.join(pathToSPE,'SPE_NormalBatches_7var_scaled.pkl'))
bad_batches_SPE = pd.read_pickle(os.path.join(pathToSPE,'SPE_FoamingBatches_7var_scaled.pkl'))
normal_batches_scores = pd.read_pickle(os.path.join(pathToSPE,'Scores_NormalBatches_7var_scaled.pkl'))
bad_batches_scores = pd.read_pickle(os.path.join(pathToSPE,'Scores_FoamingBatches_7var_scaled.pkl'))
test_batches_SPE = pd.read_pickle(os.path.join(pathToSPE,'SPE_TestBatches_7var_scaled.pkl'))
test_batches_scores = pd.read_pickle(os.path.join(pathToSPE,'Scores_TestBatches_7var_scaled.pkl'))
bad_batches_SPEVar = pd.read_pickle(os.path.join(pathToSPE,'SPEVar_FoamingBatches_7var_scaled.pkl'))
test_batches_SPEVar = pd.read_pickle(os.path.join(pathToSPE,'SPEVar_TestBatches_7var_scaled.pkl'))

#%%

alpha_online_Q = 0.9999
SPEAlpha = modular_MPCA_online.calculate_online_control_SPE(normal_batches_SPE, alpha_online_Q)
SPEAlpha.to_pickle(os.path.join(path_to_output, 'SPE_confidence_limits.pkl'))

classificationBadBatches = MPCA_functions.det_classification(bad_batches_SPE, SPEAlpha, alignedFoamingSignal)
classificationNormalBatches = MPCA_functions.det_classification(normal_batches_SPE, SPEAlpha)
classificationTestBatches = MPCA_functions.det_classification(test_batches_SPE, SPEAlpha)

confidence_matrix_Q = MPCA_functions.buildConfusionMatrix(classificationBadBatches, classificationTestBatches)
kpi_Q = MPCA_functions.calcRates(classificationBadBatches, classificationTestBatches)

confidence_matrix_Q
kpi_Q
MPCA_functions.calc_output_accuracy(confidence_matrix_Q)

#%% Plot normal batches and control limits
#==============================================================================
# for i in normalBatches:
#     fig, ax1 = plt.subplots()
#     _ = fig.suptitle('Batch %i' %i)
#     y = normal_batches_SPE[normal_batches_SPE.index == i].T
#     _ = ax1.plot(y, label = 'SPE')
#     _ = ax1.plot(SPEAlpha.loc['SPE_Control_Limit_99Perc'], label = '99% confidence limit')
#     _ = ax1.set_xlabel('time (min)')
#     _ = ax1.set_ylabel('SPE')
#     _ = ax1.set_ylim(0,500)
#     #_ = ax1.set_xlim(300,360)
#     _ = plt.legend()
#     #fig.tight_layout()
#     _ = plt.show
#==============================================================================




#%%
def plot_SPE_trajectory(batch, SPE_data, SPE_confidence_limits, foaming_trajectory = None):
    
    fig, ax1 = plt.subplots()
#    _ = fig.suptitle('Batch %i' %batch)
    
    #x = trajectorySPE.columns.values
    y = SPE_data[SPE_data.index == batch].T
    
    _ = ax1.plot(y, label = 'SPE',marker = 'o', linestyle='-', markersize = 3,
                 color = 'steelblue')
    _ = ax1.plot(SPE_confidence_limits, label = '99% confidence limit',
                 color = 'limegreen')
    _ = ax1.set_xlabel('time (min)')
    _ = ax1.set_ylabel('SPE')
    _ = ax1.set_ylim(0,200)
    #_ = ax1.set_xlim(210,235)
    
    if foaming_trajectory is not None:
        foamingTrajectory = foaming_trajectory[foaming_trajectory.batch == batch].loc[:,'LA20162.2'].reset_index(drop = True)
        foamingPoints = foamingTrajectory.index[foamingTrajectory == 1]        
        _ = ax1.axvline(foamingPoints[0], color = 'red', label = 'First foaming signal') 
        
    _ = plt.legend(prop={'size':20})
    _ = plt.show()   

#%%

if plots_on == 1:
    for batch in alignedTransformedOutlier.index:
        plot_SPE_trajectory(batch, bad_batches_SPE, SPEAlpha, alignedFoamingSignal)

#%%
for batch in alignedTransformedTest.index:
    plot_SPE_trajectory(batch, test_batches_SPE, SPEAlpha)

 
#%%
# Calculate values for confidence ellipsoid, consider last time intervall only
I = len(batchesNormalScaled.index)
def calcControl(var):
    fValue = sp_stats.f.ppf(0.99, 2, (I-2))
    control = (((var * fValue * 2 *((I**2) -1))/(I*(I-2)))**(1/2))
    return (control)

#%% Select scores of last time intervall of good batches for the creation of ellipsoids limits
scoresLimits = (normal_batches_scores.loc[:,len(batchesNormalScaled.columns.levels[0])-1])

covMatrixScores = np.cov(scoresLimits.T)
covMatrixScores[covMatrixScores < 0.0001] = 0

controlEll = pd.Series(np.diag(covMatrixScores)).apply(calcControl)
controlLimits = controlEll.to_frame()
controlLimits.index = principal_components.index.values
controlLimits.columns = ['Confidence_ellipsoids']


#%%
#Plot development of scores for foaming batches

if (plots_on == 1):

    for batch in alignedTransformedTest.index:
        scoresPlot = test_batches_scores.loc[batch].unstack()
        scoresPlot = scoresPlot[['PC1','PC2']]
    
#        x = alignedFoamingSignal[alignedFoamingSignal.batch == batch].reset_index(drop = True)
#        index = x[x['LA20162.2'] > 0].index[0]
#        
#        scoresPlot_foaming = scoresPlot[scoresPlot.index >= index]
#        scoresPlot_before = scoresPlot[scoresPlot.index < index]
        
        fig, ax = plt.subplots()
        
        xAxis = 'PC1'
        yAxis = 'PC2'
            
#        _ = plt.scatter(scoresPlot_foaming.loc[:,xAxis], 
#                    scoresPlot_foaming.loc[:,yAxis],
#                    color = 'orangered',
#                    label = 'Scores after foaming event')
#                    #label = 'foaming Batch %i' %batch)
#
#        _ = plt.scatter(scoresPlot_before.loc[:,xAxis], 
#                    scoresPlot_before.loc[:,yAxis],
#                    color = 'steelblue',
#                    label = 'Scores before foaming event')

        _ = plt.scatter(scoresPlot.loc[:,xAxis], 
                    scoresPlot.loc[:,yAxis],
                    color = 'steelblue',
                    label = 'Scores')
        
        _ = ax.set_xlim([-100,100])
        _ = ax.set_ylim([-100,100])
        
        #for i, txt in enumerate(scores.index):
        #    plt.annotate(txt, (scores.iloc[i,0],scores.iloc[i,1]))
        
#        _ = fig.suptitle('Scatterplot of first two Principal Components', fontsize = 16)
        _ = plt.xlabel(xAxis)
        _ = plt.ylabel(yAxis)
        
        _ = ax.add_patch(
            patches.Ellipse(
                (0, 0),
                width=2*(controlLimits.loc[xAxis,'Confidence_ellipsoids']),
                height =  2*(controlLimits.loc[yAxis,'Confidence_ellipsoids']),
                fill = False,
                color = 'limegreen',
                label = '99% confidence ellipse'))
        
        _ = plt.legend(prop={'size':20})
        _ = plt.show();
#%%
        
#x = normal_batches_SPE.max(axis = 1)
#x
#x[x > 1000]

#%%
#==============================================================================
# for batch in normalBatches:
#     scoresPlot = normal_batches_scores.loc[idx[batch],idx[:,'PC1':'PC2']].unstack()
# 
#     fig, ax = plt.subplots()
#     
#     xAxis = 'PC1'
#     yAxis = 'PC2'
#         
#     _ = plt.scatter(scoresPlot.loc[:,xAxis], 
#                 scoresPlot.loc[:,yAxis],
#                 color = 'b',
#                 label = 'good Batch %i' %batch)
#     
#     _ = ax.set_xlim([-150,150])
#     _ = ax.set_ylim([-150,150])
#     
#     #for i, txt in enumerate(scores.index):
#     #    plt.annotate(txt, (scores.iloc[i,0],scores.iloc[i,1]))
#     
#     _ = fig.suptitle('Scatterplot of first two Principal Components', fontsize = 16)
#     _ = plt.xlabel(xAxis)
#     _ = plt.ylabel(yAxis)
#     
#     _ = ax.add_patch(
#         patches.Ellipse(
#             (0, 0),
#             width=2*(controlLimits.loc[xAxis,'Confidence_ellipsoids']),
#             height =  2*(controlLimits.loc[yAxis,'Confidence_ellipsoids']),
#             fill = False,
#             color = 'red'))
#     
#     _ = plt.legend()
#     _ = plt.show();
#==============================================================================

#%%
# Calculate Hotelling statistic for online algorithm
def calculateDOnline(scores, I, R, invMatrix):    
    result = ((np.dot(np.dot(scores.values, invMatrix),(scores.T).values))*I*(I-R))/(R*((I**2)-1))
    return (result)

alpha = 0.99
I = len(batchesNormalScaled.index)
R = len(principal_components.index)
inverseMatrix = np.linalg.inv(covMatrix)
controlValue = sp_stats.f.ppf(alpha, (R), (I-R))
#%%
for batch in alignedTransformedOutlier.index:
    
    foamP = MPCA_functions.foamingPoint(alignedFoamingSignal, batch)
    x = bad_batches_scores.loc[batch].unstack()
    x = x[list(principal_components.index)]
    values = x
    Dvalues = values.apply(calculateDOnline, axis = 1, args = (I,R,inverseMatrix))      
    
    fig, ax1 = plt.subplots()
#    _ = fig.suptitle('Batch %i' %batch)
    
    x = Dvalues.index
    y = Dvalues.values
    
    _ = ax1.plot(x, y, label = 'D',marker = 'o', linestyle='-', markersize = 3,
                 color = 'steelblue')
    _ = ax1.set_xlabel('time (min)')
    _ = ax1.set_ylabel('D')
    #_ = ax1.set_ylim(0,500)
    #_ = ax1.set_xlim(300,360)
    _ = ax1.axhline(controlValue, color = 'limegreen', label = '99% control value')
    _ = ax1.axvline(foamP, color = 'orangered', label = 'First foaming signal')
    _ = plt.legend(prop={'size':20})
    _ = plt.show
    
    
#%%

for batch in random.sample(list(alignedTransformedTest.index), 20):
    
    x = test_batches_scores.loc[batch].unstack()
    x = x[list(principal_components.index)]
    values = x
    Dvalues = values.apply(calculateDOnline, axis = 1, args = (I,R,inverseMatrix))      
    
    fig, ax1 = plt.subplots()
#    _ = fig.suptitle('Batch %i' %batch)
    
    x = Dvalues.index
    y = Dvalues.values
    
    _ = ax1.plot(x, y, label = 'D',marker = 'o', linestyle='-', markersize = 3,
                 color = 'steelblue')
    _ = ax1.set_xlabel('time (min)')
    _ = ax1.set_ylabel('D')
    _ = ax1.set_ylim(0,3)
    #_ = ax1.set_xlim(300,360)
    _ = ax1.axhline(controlValue, color = 'limegreen', label = '99% control value')
#    _ = ax1.axvline(foamP, color = 'orangered', label = 'First foaming signal')
    _ = plt.legend(prop={'size':20})
    _ = plt.show
    
    

    
#%%
"""
Calculate results with the use of D-measure

"""

alpha = 0.9999
I = len(batchesNormalScaled.index)
R = len(principal_components.index)
inverseMatrix = np.linalg.inv(covMatrix)
controlValue = sp_stats.f.ppf(alpha, (R), (I-R))
classification = {}

for batch in alignedTransformedOutlier.index: 
    foamP = MPCA_functions.foamingPoint(alignedFoamingSignal, batch)
    x = bad_batches_scores.loc[batch].unstack()
    x = x[list(principal_components.index)]
    values = x[x.index<foamP-1]
    Dvalues = values.apply(calculateDOnline, axis = 1, args = (I,R,inverseMatrix))  
    binTable = Dvalues > controlValue
    sumFailures = binTable.sum()
    if (sumFailures > 0):
        boolean = True
    else:
        boolean = False 
    classification[batch] = boolean
    predFaults = sum(classification.values())
    predNoFaults = len(classification) - predFaults
    
badBatchesDClassification = (predFaults, predNoFaults)

classification = {}

for batch in test_batches_scores.index: 
    x = test_batches_scores.loc[batch].unstack()
    x = x[list(principal_components.index)]
    values = x
    Dvalues = values.apply(calculateDOnline, axis = 1, args = (I,R,inverseMatrix))  
    binTable = Dvalues > controlValue
    sumFailures = binTable.sum()
    if (sumFailures > 0):
        boolean = True
    else:
        boolean = False 
    classification[batch] = boolean
    predFaults = sum(classification.values())
    predNoFaults = len(classification) - predFaults
    
testBatchesDClassification = (predFaults, predNoFaults)

confMatrixD = MPCA_functions.buildConfusionMatrix(badBatchesDClassification, testBatchesDClassification)
kpiD = MPCA_functions.calcRates(badBatchesDClassification, testBatchesDClassification)
confMatrixD
kpiD
MPCA_functions.calc_output_accuracy(confMatrixD)





#%% Plot contribution of each variable to SPE at certain time t

dict_foaming_batches = {90:245, 92: 180, 100: 521, 102:248, 103:500, 105: 491, 
                        107:310, 133:71, 165:70, 171:245, 185:68, 256:71, 257:241, 
                        259:239, 260:242, 268:222, 284:73}
for key, value in dict_foaming_batches.items():
    MPCA_functions.barplotVarSPE(bad_batches_SPEVar, key, value)
    plot_SPE_trajectory(key, bad_batches_SPE, SPEAlpha, alignedFoamingSignal)

#%%

# With this cell, the classifier can be judged according to the classification for Q

pd.set_option('precision',15)
alphas = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99, 0.995, 0.999, 0.9999, 0.99999,
          0.999999, 0.9999999, 0.99999999, 0.999999999, 0.9999999999, 0.99999999999,
          0.999999999999, 0.9999999999999, 0.99999999999999, 0.999999999999999,
           0.9999999999999999, 0.99999999999999999]

#%%
result_ROC_curve = pd.DataFrame()

for i in range(5000,10000):
    alpha_online_Q = i/10000
    SPEAlpha = modular_MPCA_online.calculate_online_control_SPE(normal_batches_SPE, alpha_online_Q)
    
    classificationBadBatches = MPCA_functions.det_classification(bad_batches_SPE, SPEAlpha, alignedFoamingSignal)
    classificationNormalBatches = MPCA_functions.det_classification(normal_batches_SPE, SPEAlpha)
    classificationTestBatches = MPCA_functions.det_classification(test_batches_SPE, SPEAlpha)
    
    confidence_matrix_Q = MPCA_functions.buildConfusionMatrix(classificationBadBatches, classificationTestBatches)
    kpi_Q = MPCA_functions.calcRates(classificationBadBatches, classificationTestBatches)

    result = MPCA_functions.return_output_ROC(confidence_matrix_Q, kpi_Q, alpha_online_Q)
    result_ROC_curve = result_ROC_curve.append(result, ignore_index = True)
      
print(result_ROC_curve)
auc_SPE = sk.metrics.auc(result_ROC_curve['FPR'], result_ROC_curve['TPR'])

#%%
def plot_ROC_curve(classifier_results, auc):
    plt.figure()
    lw = 2
    plt.plot(classifier_results['FPR'], classifier_results['TPR'], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", prop={'size':20})
    plt.show()

#%%
plot_ROC_curve(result_ROC_curve, auc_SPE)

#%%

# With this cell, the classifier can be judged according to the classification for D

result_ROC_curve_D = pd.DataFrame()

for alpha in alphas:
    I = len(batchesNormalScaled.index)
    R = len(principal_components.index)
    inverseMatrix = np.linalg.inv(covMatrix)
    controlValue = sp_stats.f.ppf(alpha, (R), (I-R))
    classification = {}
    
    for batch in alignedTransformedOutlier.index: 
        foamP = MPCA_functions.foamingPoint(alignedFoamingSignal, batch)
        x = bad_batches_scores.loc[batch].unstack()
        x = x[list(principal_components.index)]
        values = x[x.index<foamP-1]
        Dvalues = values.apply(calculateDOnline, axis = 1, args = (I,R,inverseMatrix))  
        binTable = Dvalues > controlValue
        sumFailures = binTable.sum()
        if (sumFailures > 0):
            boolean = True
        else:
            boolean = False 
        classification[batch] = boolean
        predFaults = sum(classification.values())
        predNoFaults = len(classification) - predFaults
        
    badBatchesDClassification = (predFaults, predNoFaults)
    
    classification = {}
    
    for batch in test_batches_scores.index: 
        x = test_batches_scores.loc[batch].unstack()
        x = x[list(principal_components.index)]
        values = x
        Dvalues = values.apply(calculateDOnline, axis = 1, args = (I,R,inverseMatrix))  
        binTable = Dvalues > controlValue
        sumFailures = binTable.sum()
        if (sumFailures > 0):
            boolean = True
        else:
            boolean = False 
        classification[batch] = boolean
        predFaults = sum(classification.values())
        predNoFaults = len(classification) - predFaults
        
    testBatchesDClassification = (predFaults, predNoFaults)
    
    confidence_matrix_D = MPCA_functions.buildConfusionMatrix(badBatchesDClassification, testBatchesDClassification)
    kpi_D = MPCA_functions.calcRates(badBatchesDClassification, testBatchesDClassification)
    result = MPCA_functions.return_output_ROC(confidence_matrix_D, kpi_D, alpha)
    result_ROC_curve_D = result_ROC_curve_D.append(result, ignore_index = True)

print(result_ROC_curve_D)
auc_D = sk.metrics.auc(result_ROC_curve_D['FPR'], result_ROC_curve_D['TPR'])



#%%

plot_ROC_curve(result_ROC_curve_D, auc_D)














#%%

for batch in alignedTransformedTest.index:
    t = 221
    if ((test_batches_SPEVar.loc[batch].loc[t].sum()) > 100):    
        MPCA_functions.barplotVarSPE(test_batches_SPEVar, batch, t)

#%%
batches_list = []
t_list = []
var_list = []
for batch in alignedTransformedTest.index:
    binTable = (test_batches_SPE[test_batches_SPE.index == batch].iloc[0,:] > SPEAlpha)
    if (True in binTable.values):        
        t = binTable[binTable == True].index[0] 
        #MPCA_functions.barplotVarSPE(test_BatchesSPEVar, batch, t)
        var_list.append((test_batches_SPEVar.loc[batch].loc[t]).argmax())        
        batches_list.append(batch)
        t_list.append(t)
#%%
        
var_time_limit = pd.DataFrame([batches_list, t_list, var_list])
batches_list

#%%

batches_list = []
t_list = []
var_list = []
for batch in alignedTransformedTest.index:
    binTable = (test_batches_SPE[test_batches_SPE.index == batch].iloc[0,:] > SPEAlpha)
    if (True in binTable.values):        
        t = binTable[binTable == True].index
        for i in t:            
            var_list.append((test_batches_SPEVar.loc[batch].loc[i]).argmax())        
            batches_list.append(batch)
            t_list.append(i)
        
#%%

df_x = pd.DataFrame([var_list, t_list, batches_list]).T
df_x.columns = ['var', 't', 'batch' ]
df_x[(df_x['var'] == 'PC20162') & (df_x['batch'] > 245) ].batch.unique()

#%%
set(batches_list)

#%%        
var_unique = set(var_list)

for i in var_unique:
    x = var_list.count(i)
    ratio = x/len(var_list)
    print(i, ratio)
    