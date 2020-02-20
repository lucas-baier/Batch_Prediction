# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:38:56 2017

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
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import MCPA_functions
from sklearn import preprocessing
from sklearn import decomposition

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
pathToCrossValidation="C://project-data/my-data-sets/cross_validation/"

#%%
alignedTransformedNormal = pd.read_pickle(os.path.join(pathToDataAligned,'alignedTransformedNormal.pkl'))

#removeVar = ['F20103B', 'F20107A', 'F20107B', 'F20107C', 'F20113',
#       'FC20101','FC20101_Y','FC20101_W', 'FC20102', 'FC20102_W',
#       'FC20102_Y', 'FC20103A','FC20103A_Y','FC20103A_W', 'FC20104','FC20104_W',
#       'FC20104_Y', 'FC20105', 'FC20105_W', 'FC20105_Y',
#       'FC20110','FC20110_W','FC20110_Y','FC20160', 'FC20160_Y','FC20160_W',
#       'FC20192', 'FC20192_W', 'FC20192_Y','FC20197','FC20197_W','FC20197_Y',
#       'FQ20101', 'FQ20102',
#       'FQ20103A', 'FQ20104', 'FQ20105', 'FQ20107A', 'FQ20107C', 'FQ20113',
#       'FQ20141']            
varArray = ['PC20162', 'FC20144', 'P20160', 'TC20171A', 'TC20172A', 'FC20102', 'E20161']

#varArray = ['PC20162', 'FC20144', 'P20160', 'TC20171A', 'TC20172A', 'FC20102', 'E20161', 
#            'FC20161_W', 'TC20172A_W', 'FC20161', 'PC20162_W', 'F20165', 'F20141', 'FC20144_Y'] 

alignedTransformedNormal = alignedTransformedNormal.loc[:,idx[:,varArray]]
columnsValues = pd.MultiIndex.from_tuples(alignedTransformedNormal.columns.values, names=['time', 'variables'])

#%%
dataScaled = pd.DataFrame(preprocessing.scale(alignedTransformedNormal), 
                                        index = alignedTransformedNormal.index, 
                                        columns = columnsValues)
                                        

del alignedTransformedNormal
  
#%%
    
maxNoComponents = 30

#%%

def calculateSPECross (scores, pComp, data, columns):
    projection = pd.DataFrame(np.dot(scores, pComp),
                              index = data.index,
                              columns = columns)
    SPE = ((data.loc[idx[:], idx[:,:]] - projection.loc[idx[:], idx[:,:]])**2).sum(axis = 1)
    return (SPE)  

#%%  
Press = {}

for noComponents in range(1,(maxNoComponents+1)):
    SPE = 0
    columnnamesPC = ["PC%i" %s for s in range(1,noComponents + 1)]
    for batch in (dataScaled.index):
    
        batchesForModel = dataScaled[np.logical_not(dataScaled.index == batch)]   
        batchForPrediction = dataScaled[dataScaled.index == batch]
    
        pca = decomposition.PCA(n_components = noComponents)
        pca.fit(batchesForModel)
    
        scoresBatch = MCPA_functions.calculateScores(batchForPrediction, pca.components_, columnnamesPC)
        SPEBatch = calculateSPECross(scoresBatch, pca.components_, batchForPrediction, columnsValues)
        SPE = SPE + SPEBatch.values
    
    Press[noComponents] = SPE
    
PressData = pd.DataFrame.from_records([Press], index = ['Sum of Squares'])
PressData.to_pickle(os.path.join(pathToCrossValidation, 'Press_7var.pkl'))

#%%

def calculateSPECrossAll (scores, pComp, data, columns, batch):
    projection = pd.DataFrame(np.dot(scores, pComp),
                              index = data.index,
                              columns = columns)
    SPE = ((data.loc[idx[:], idx[:,:]] - projection.loc[idx[:], idx[:,:]])**2).sum(axis = 1)
    return (SPE)  


#%%
RSS = {}

for noComponents in range(1,(maxNoComponents+1)):
    pca = decomposition.PCA(n_components = noComponents)
    pca.fit(dataScaled)
    
    columnnamesPC = ["PC%i" %s for s in range(1,pca.n_components_ + 1)]
    
    scoresBatches = MCPA_functions.calculateScores(dataScaled, pca.components_, columnnamesPC)
    SPEBatches = calculateSPECrossAll(scoresBatches, pca.components_, dataScaled, columnsValues, batch = dataScaled.index)
    RSS[noComponents] = SPEBatches.sum()
    
RSSData = pd.DataFrame.from_records([RSS], index = ['Sum of Squares'])
RSSData.to_pickle(os.path.join(pathToCrossValidation, 'RSS_7var.pkl'))

#%% Calculate ratio R
R = {}
for noComp in range(2,(maxNoComponents+1)):
    R[noComp] = Press[noComp] / RSS[noComp - 1]

RData = pd.DataFrame.from_records([R], index = ['R value'])
RData.to_pickle(os.path.join(pathToCrossValidation, 'R_7var.pkl'))  
                        
#%% Calculate ratio by Krzanowski

DM = {}
DR = {}
W = {}

I = len(dataScaled.index)
J = len(dataScaled.columns.levels[1])
K = len(dataScaled.columns.levels[0])      

for noComp in range(2,(maxNoComponents+1)):

    DM[noComp] = I + J*K - 2*noComp      
    sumEnd = sum([(I+J*K-2*i) for i in range(1, noComp+1)])   
    DR[noComp] = J*K*(I-1) - sumEnd
    
    W[noComp] = ((Press[noComp-1] - Press[noComp])/DM[noComp])/(Press[noComp]/DR[noComp])

WData = pd.DataFrame.from_records([W], index = ['W value'])
WData.to_pickle(os.path.join(pathToCrossValidation, 'W_7var.pkl'))             

#%%
R = pd.read_pickle(os.path.join(pathToCrossValidation, 'R_7var.pkl'))
W =  pd.read_pickle(os.path.join(pathToCrossValidation, 'W_7var.pkl'))
#%%
RSS =  pd.read_pickle(os.path.join(pathToCrossValidation, 'RSS_7var.pkl'))
Press =  pd.read_pickle(os.path.join(pathToCrossValidation, 'Press_7var.pkl'))
#%%
listPress = []
for i in Press.values[0].tolist():
    listPress.append(i[0])
#%%

W    
#%%
fig,ax = plt.subplots()
plt.plot(range(1,(maxNoComponents+1)),listPress, color = 'steelblue')
#ax.set_xlim([5,15])
ax.set_xlabel('Number of components', size = 20)
ax.tick_params(axis='x', which='major', labelsize =16)
ax.tick_params(axis='y', which='major', labelsize =16)
plt.show()
#%%
w_list = []
for i in range(29):
    value = W.iloc[:,i].values[0][0]
    w_list.append(value)
    
#%%
len(w_list)
len(range(2,31))
   #%% 
fig, ax = plt.subplots()
plt.plot(range(2,31),w_list, color = 'steelblue')
ax.set_xlabel('Number of components', size = 20)
ax.tick_params(axis='x', which='major', labelsize =16)
ax.tick_params(axis='y', which='major', labelsize =16)
plt.show()

#%%
R.T

#use 10 PCs due to ellbow criteria and W Value with 7 variables