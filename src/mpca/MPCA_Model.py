# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:14:11 2017

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

# import other MPCA packages
import MPCA_functions
import modular_MPCA_main
import importlib
importlib.reload(MPCA_functions)
importlib.reload(modular_MPCA_main)

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
#%%
#
# PATH TO DATA - MODIFY IT APPROPRIATELY!!!
#
pathToDataSets="C:/project-data/my-data-sets/aligned"
pathToDataSets_new="C:/project-data/my-data-sets/data_new_processed/"
pathToOutput="C://project-data/my-data-sets/SPE_trajectories/"

alignedTransformed = pd.read_pickle(os.path.join(pathToDataSets_new,'alignedBatches_reshaped.pkl'))
alignedFoamingSignal = pd.read_csv(os.path.join(pathToDataSets_new,'alignedFoamingSignal_scaled.csv'))
#Show plots, 1 = yes, 0 = no
plots_on = 1

# Returns index of outlier batches
allOutliers, badBatchesFoaming = MPCA_functions.abnormalBatchesIndex(alignedTransformed, alignedFoamingSignal)
test_set_index = MPCA_functions.test_index_old_batches
test_index_new = MPCA_functions.test_index_new_batches
test_set_index = test_set_index + test_index_new

# Set no of components to 10
no_components = 10


#%%                   
#Variables to keep in process
          
varArray = [ 'E20161', 'FC20144', 'P20160','PC20162', 'TC20171A',
            'FC20102', 'TC20172A'] #'FC20161'] # 'TC20172A',


alignedTransformed = alignedTransformed.loc[:,idx[:,varArray]]
columnsValues = pd.MultiIndex.from_tuples(alignedTransformed.columns.values, names=['time', 'variables'])

#%%

# Separate data into normal (training) batches, test batches and outlier batches
alignedTransformedNormal, alignedTransformedTest, alignedTransformedOutlier, columnsValues = modular_MPCA_main.split_data(
    varArray, alignedTransformed, allOutliers, badBatchesFoaming, test_set_index)

#%%

#  Standardization of batches with the use of the 50 batches before

#normal_batches_scaled, test_batches_scaled, outlier_batches_scaled = MPCA_functions.standardize_data_current(
#    alignedTransformedNormal, alignedTransformedTest, alignedTransformedOutlier, columnsValues, 50)

#Standardize data (every column) with mean and standard deviation
normal_batches_scaled, test_batches_scaled, outlier_batches_scaled, mean, std = modular_MPCA_main.standardize_data(
    alignedTransformedNormal, alignedTransformedTest, alignedTransformedOutlier, columnsValues)

mean_columnwise = alignedTransformedNormal.mean(axis = 0)
std_columnwise = alignedTransformedNormal.std(axis = 0)

#%%
#Build full model

allComponents = len(alignedTransformedNormal.index)
pca = decomposition.PCA(n_components = allComponents)
pca.fit(normal_batches_scaled)

eigenvaluesAll = pca.explained_variance_
#pca.explained_variance_ratio_
np.sum(pca.explained_variance_ratio_)

#Build MPCA model with predetermined number of components
pca = decomposition.PCA(n_components = no_components)
pca.fit(normal_batches_scaled)

# Show sum of explained variance with model
np.sum(pca.explained_variance_ratio_)

columnnamesPC = ["PC%i" %s for s in range(1,pca.n_components_ + 1)]
prinComponents = pd.DataFrame(pca.components_, index = columnnamesPC, columns = columnsValues)


#%%
# Compute scores for training, test and foaming batches
normal_batches_scores = MPCA_functions.calculateScores(normal_batches_scaled, pca.components_, columnnamesPC)
test_batches_scores = MPCA_functions.calculateScores(test_batches_scaled, pca.components_, columnnamesPC)  
outlier_batches_scores = MPCA_functions.calculateScores(outlier_batches_scaled, pca.components_, columnnamesPC)

#%%
# Plot the scatterplot for the scores
if (plots_on == 1):    
    MPCA_functions.plot_scatterplot_scores(normal_batches_scores,
                                           outlier_batches_scores, 
                                           test_batches_scores,
                                           'PC1', 
                                           'PC2',
                                           outlier_batches_scores)

#%%

# Calculate the Residual Matrix E for all batches

E = MPCA_functions.calcErrorMat(normal_batches_scores, pca.components_, normal_batches_scaled, columnsValues)
ResidualMatrixSquared = E**2

ResidualMatrixOutlierSquared = (MPCA_functions.calcErrorMat(outlier_batches_scores, 
                                                            pca.components_, 
                                                            outlier_batches_scaled, 
                                                            columnsValues)**2)

ResidualMatrixTestSquared = (MPCA_functions.calcErrorMat(test_batches_scores, 
                                                         pca.components_,
                                                         test_batches_scaled,
                                                         columnsValues)**2)

# Calculate Q for all batches by summing up the Residual Matrix

Q = pd.DataFrame(ResidualMatrixSquared.sum(1), columns = ['SumOfSquares'])
QOutlier = pd.DataFrame(ResidualMatrixOutlierSquared.sum(1), columns = ['SumOfSquares'])
QFoaming = QOutlier
#QSteam = QOutlier[QOutlier.index.isin(scenarioSteamReduction)]
QTest = pd.DataFrame(ResidualMatrixTestSquared.sum(1), columns = ['SumOfSquares'])
#QFoaming.sort_values(by = 'SumOfSquares', ascending = False)


#%% Calculate confidence limits for SPE, see Nomikos and MacGregor 1995

alphaQ = 0.99

theta = dict()
for i in range(1,4):
    x = eigenvaluesAll[no_components:]
    theta[i] = np.power(x,i).sum()
 
h0 = 1- ((2*theta[1]*theta[3])/(3*(np.power(theta[2],2))))
cAlpha = pd.Series([sp.stats.norm.ppf(alphaQ), sp.stats.norm.ppf(0.950)])
QAlpha = theta[1]*np.power((((h0*cAlpha*np.power(2*theta[2],0.5))/(theta[1]))+
                           1+((theta[2]*h0*(h0-1))/(np.power(theta[1],2)))),(1/h0))


#%%
# Plot Q contribution plot for anonymous variable names
# Other function can be found in MPCA_functions script
    
def barplot_Q_model (batches_Q_per_variable, batch):
    data = batches_Q_per_variable[batches_Q_per_variable.index == batch]
    x = range(data.shape[1])
    y = data.values[0].tolist()
    labels = ['Variable 1', 'Variable 2', 'Variable 3', 'Variable 4', 'Variable 5', 'Variable 6', 'Variable 7', ]
    
    fig, ax = plt.subplots()
    
    _ = plt.bar(x, y, align='center', color = 'steelblue')
    _ = plt.xticks(x, labels)
    #_ = ax.set_ylim([0,500])
    
#    _ = plt.xlabel('Variables')
    _ = plt.ylabel('Q')
#    _ = plt.title('Overview on prediction error per variable, batch %i' %(batch))
    _ = plt.legend()
    
    _ = plt.show();
    
    return(fig)

#%%
# Select the Residual Matrix and plot Q contribution plot

df_plot = ResidualMatrixSquared

for batch in  [255]:    
    Q_values =MPCA_functions.Q_contribution_plot_values(df_plot)    
    barplot_Q_model(Q_values, batch)




#%%
# Function that plots the orignal trajectory of a variable and the corresponding reconstruction
# with the use of score and principal components

def plot_trajectories_and_projection (data, batch, scores, var, principal_components, columns_values):
    time_series = data.loc[batch].unstack().loc[:,var] 
                    
    time_series_projection = pd.DataFrame(np.dot(scores[scores.index == batch], principal_components),
                                          index = [batch],
                                          columns = columns_values)
    columns = time_series_projection.stack().T.columns.droplevel()
    time_series_var_projection = time_series_projection.stack().T
    time_series_var_projection.columns = columns
    x = time_series_var_projection.loc[:,var]                                  
    
    ax, fig = plt.subplots()
    _ = plt.plot(time_series, label = 'Original batch data', color = 'grey')
    _ = plt.plot(x, label = 'Projection', color = 'steelblue')
    _ = plt.legend(prop={'size':20})
    return (fig)

#%%
#Plot the reconstructed variable trajectory and analyze the which time intervals 
#contribute the most to the Q value

index  = Q[Q.SumOfSquares > QAlpha[0]].index
for batch_analysis in [31]:
    var_analysis = 'E20161'
        
#    MPCA_functions.SPE_per_time_interval(ResidualMatrixTestSquared, batch_analysis, var_analysis)
    plot_trajectories_and_projection(normal_batches_scaled, batch_analysis, normal_batches_scores, var_analysis,
                                     prinComponents, columnsValues)   
                 
#%%
# Plot bar chart of Q values
if (plots_on == 1):
    MPCA_functions.plot_barchart_Q(Q, QFoaming, QTest, QAlpha)

#%% Analysis of Q and determine post classification result

resultAccuracy = pd.DataFrame ({'Fault' : pd.Series([len(QTest[QTest.SumOfSquares < QAlpha[0]].index), len(QFoaming[QFoaming.SumOfSquares < QAlpha[0]].index)], index = ['NoFault', 'Fault']),
                                'NoFault' : pd.Series([len(QTest[QTest.SumOfSquares > QAlpha[0]].index), len(QFoaming[QFoaming.SumOfSquares > QAlpha[0]].index)], index = ['NoFault', 'Fault'])})
indexConfusion = pd.MultiIndex.from_tuples( [('Actual', 'NoFault'),('Actual','Fault')])
columnsConfusion = pd.MultiIndex.from_tuples( [('Predicted', 'NoFault'),('Predicted','Fault')])
resultAccuracy.index = indexConfusion
resultAccuracy.columns = columnsConfusion
resultAccuracy

#%% Calculate hotelling statistic, defined in Nomikos, MacGregor 1995

alphaD = 0.99

covMatrix = np.cov(normal_batches_scores.T)
covMatrix[covMatrix < 0.00001] = 0
c = pd.DataFrame(covMatrix)
#c.to_pickle(os.path.join(pathToDataSets,'covMatrixReference.pkl'))
invCovMatrix = np.linalg.inv(covMatrix)

# Function which computes the D statistic per row 
def calculateD(scores, I, invMatrix):    
    result = ((np.dot(np.dot(scores.values, invMatrix),(scores.T).values))*I)/((I-1)**2)
    return (result)
     
DS = normal_batches_scores.apply(calculateD,
                                 args = (len(normal_batches_scores.index),
                                         invCovMatrix),
                                         axis = 1)
DSTest = test_batches_scores.apply(calculateD, 
                                   args = (len(test_batches_scores.index),
                                           invCovMatrix),
                                           axis = 1)
DSFoaming = outlier_batches_scores.apply(calculateD, 
                                         args = (len(outlier_batches_scores.index),
                                                 invCovMatrix),
                                                 axis = 1)

# Compute control value
I = len(normal_batches_scores.index)
R = no_components
nominator = ((R/(I-R-1))*sp.stats.f.ppf(alphaD, (R), (I-R-1)))
denominator = (1+(R/(I-R-1))*sp.stats.f.ppf(alphaD, (R), (I-R-1)))
hotellingStat = nominator/denominator

#%%
# Plot bar chart for D statistic per batch
if (plots_on == 1):

    fig, ax = plt.subplots()
    
    #bar_width = 0.35
    
    plt.bar(DS.index, 
            DS, 
            color = 'lightsteelblue', 
            label = 'Training set')
            
    plt.bar(DSTest.index, 
            DSTest, 
            color = 'darkblue',
            label = 'Test set - normal batches') 
            
    plt.bar(DSFoaming.index, 
            DSFoaming, 
            color = 'orangered', 
            label = 'Test set - foaming batches')
    
    ax.axhline(hotellingStat, label = '99% - confidence limit', color = 'black')
    
    ax.set_ylim([0,3])
    ax.set_xlim([0,150])
    
    plt.xlabel('Batches')
    plt.ylabel('D')
#    plt.title('Hotelling statistic')
    plt.legend(prop={'size':20})
    
    plt.show();

#%% Calculate Confusion Matrix for Hotelling statistic
resultAccuracyD = pd.DataFrame ({'Fault' : pd.Series([len(DSTest[DSTest < hotellingStat]), len(DSFoaming[DSFoaming < hotellingStat])], index = ['NoFault', 'Fault']),
                                'NoFault' : pd.Series([len(DSTest[DSTest > hotellingStat]), len(DSFoaming[DSFoaming > hotellingStat])], index = ['NoFault', 'Fault'])})
indexConfusion = pd.MultiIndex.from_tuples( [('Actual', 'NoFault'),('Actual','Fault')])
columnsConfusion = pd.MultiIndex.from_tuples( [('Predicted', 'NoFault'),('Predicted','Fault')])
resultAccuracyD.index = indexConfusion
resultAccuracyD.columns = columnsConfusion
resultAccuracyD

#%% Calculate confidence ellipsoids for score scatterplots
alpha = 0.8

def calculateControl(var, betaVar, I):
    return (((var * betaVar * ((I-1)**2))/I)**(1/2))

R = 2
nominator = ((R/(I-R-1))*sp.stats.f.ppf(alpha, (R), (I-R-1)))
denominator = (1+(R/(I-R-1))*sp.stats.f.ppf(alpha, (R), (I-R-1)))
betaVar = nominator/denominator
control = pd.Series(np.diagonal(covMatrix)).apply(calculateControl, args = (betaVar, I))
controlLimits = control.to_frame()
controlLimits.index = columnnamesPC
controlLimits.columns = ['Confidence_ellipsoids']

#%%

outlier_batches_scores = outlier_batches_scores[outlier_batches_scores.index != 107]

outlier_batches_scores_steam = outlier_batches_scores[outlier_batches_scores.index.isin(MPCA_functions.scenarioSteam)]
outlier_batches_scores_post_steam = outlier_batches_scores[outlier_batches_scores.index.isin(MPCA_functions.scenarioPostSteam)]


#%%

# Scatterplot of scores where one can differentiate between foaming sceanrios
if (plots_on == 1):

    fig, ax = plt.subplots()
    
    xAxis = 'PC1'
    yAxis = 'PC2'
        
    plt.scatter(normal_batches_scores.loc[:,xAxis], 
                normal_batches_scores.loc[:,yAxis],
                color = 'lightsteelblue',
                label = 'Training set')

    plt.scatter(test_batches_scores.loc[:,xAxis], 
                test_batches_scores.loc[:,yAxis],
                color = 'cornflowerblue',
                label = 'Test set - normal batches')       

#    plt.scatter(outlier_batches_scores.loc[:,xAxis], 
#                outlier_batches_scores.loc[:,yAxis],
#                color = 'orangered',
#                label = 'Test set - foaming batches')   
                
    plt.scatter(outlier_batches_scores_steam.loc[:,xAxis], 
                outlier_batches_scores_steam.loc[:,yAxis],
                color = 'orangered',
                label = 'Test set - foaming in phase 2')
                
    plt.scatter(outlier_batches_scores_post_steam.loc[:,xAxis], 
                outlier_batches_scores_post_steam.loc[:,yAxis],
                color = 'sienna',
                label = 'Test set - foaming in phase 3')
                
         
    # Add batch number to each batch
    #for i, txt in enumerate(scores.index):
    #    plt.annotate(txt, (scores.iloc[i,0],scores.iloc[i,1]))
    
    #fig.suptitle('Scatterplot of scores on first two Principal Components', fontsize = 16)
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)

# Add control ellipse to plot
    
#    ax.add_patch(
#        patches.Ellipse(
#            (0, 0),
#            width=2*(controlLimits.loc[xAxis,'Confidence_ellipsoids']),
#            height =  2*(controlLimits.loc[yAxis,'Confidence_ellipsoids']),
#            fill = False,
#            color = 'red'
#        )
#    )
    
    plt.legend(prop={'size':20})
    
    plt.show();

#%% Calcualte control limits for SPE per time interval based on error matrix
#==============================================================================
# 
# meanNormal = (ResidualMatrixSquared.mean()).mean()
# varNormal = (ResidualMatrixSquared.var(axis = 0)).mean()
# g = varNormal/(2*meanNormal)
# h = (2*(meanNormal**2))/(varNormal)
# 
# SPEAlpha95 = varNormal/(2*meanNormal)*sp.stats.chi2.ppf(q = 0.95, df = ((2*meanNormal**2)/(varNormal)))
# SPEAlpha999 = varNormal/(2*meanNormal)*sp.stats.chi2.ppf(q = 0.999, df = ((2*meanNormal**2)/(varNormal)))
# 
#==============================================================================
#%%
# Plot SPE trajectory per time interval for foaming batches

#==============================================================================
# for batch in badBatchesFoaming:
# 
#     QTime = ResidualMatrixOutlierSquared[ResidualMatrixOutlierSquared.index == batch].stack(0)
#     QTime['Sum'] = QTime.sum(axis = 1)
#     
#     foamP = MPCA_functions.foamingPoint(alignedFoamingSignal, batch)
#     
#     fig, ax = plt.subplots()
#     
#     _ = plt.plot(QTime.index.levels[1].values,
#             QTime.loc[:,'Sum'].values, 
#             color = 'b',
#             label = 'SPE')
#             
#     _ = ax.axvline(foamP, color = 'red', label = 'First foaming signal')
#     _ = ax.axhline(SPEAlpha999, color = 'black', label = 'Confidence limit')
#     
#     _ = fig.suptitle('SPE trajectory for batch %i' %batch, fontsize = 16)
#     _ = plt.xlabel('time')
#     _ = plt.ylabel('SPE')
#     
#     _ = plt.show();
# 
#==============================================================================

#%%
#==============================================================================
# for batch in normalBatches:
# 
#     QTime = ResidualMatrixSquared[ResidualMatrixSquared.index == batch].stack(0)
#     QTime['Sum'] = QTime.sum(axis = 1)
#     
#     fig, ax = plt.subplots()
#     
#     _ = plt.plot(QTime.index.levels[1].values,
#             QTime.loc[:,'Sum'].values, 
#             color = 'b',
#             label = 'SPE')
#             
#     _ = ax.axhline(SPEAlpha999, color = 'black', label = 'Confidence limit')
#     
#     _ = fig.suptitle('SPE trajectory for batch %i' %batch, fontsize = 16)
#     _ = plt.xlabel('time')
#     _ = plt.ylabel('SPE')
#     
#     _ = plt.show();
#==============================================================================
    
#%% Plot prediction errors in the individual process variables
#==============================================================================
# for batch in badBatchesFoaming:
#     foamP = MPCA_functions.foamingPoint(alignedFoamingSignal, batch)
#     QTime = ResidualMatrixOutlierSquared[ResidualMatrixOutlierSquared.index == batch].stack(0)
#     x = QTime[QTime.index.levels[1] == (foamP-10)]
#     
#     N = x.shape[1]
#     p = range(N)
#     y = x.values[0].tolist()
#     labels = (x.columns.values).tolist()
#     
#     fig, ax = plt.subplots()
#     
#     _ = plt.bar(p, y, align='center', color = 'b')
#     _ = plt.xticks(p, labels)
#     #_ = ax.set_ylim([0,500])
#     
#     _ = plt.xlabel('Variables')
#     _ = plt.ylabel('Q')
#     _ = plt.title('Overview on prediction error per variable, batch %i, t = %i' %(batch, foamP))
#     _ = plt.legend()
#     
#     _ = plt.show();
#==============================================================================
