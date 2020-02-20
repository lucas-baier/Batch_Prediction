# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:33:44 2017

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
import pandas as pd
import os.path
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import cluster
from MPCA_functions import standardizeInput, copyDevVector, calculateScores

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

#%%

#alignedTransformedNormal = pd.read_pickle(os.path.join(pathToDataAligned,'alignedTransformedNormal.pkl'))
alignedTransformedOutlier = pd.read_pickle(os.path.join(pathToDataAligned,'alignedTransformedOutlier.pkl'))
alignedFoamingSignal = pd.read_csv(os.path.join(pathToDataAligned,'alignedFoamingSignal.csv'))

prinComponents = pd.read_pickle(os.path.join(pathToDataAligned, 'PrincipalComponents.pkl'))

mean_columnwise = pd.read_pickle(os.path.join(pathToDataAligned,'mean_columnwise.pkl'))
std_columnwise = pd.read_pickle(os.path.join(pathToDataAligned,'std_columnwise.pkl'))

badBatchesSPE = pd.read_pickle(os.path.join(pathToSPE,'SPE_FoamingBatches_7var.pkl'))

badBatchesFoaming = alignedTransformedOutlier.index[alignedTransformedOutlier.index.isin(list( set(scenarioSteam) | set(scenarioPostSteam)))]

columnsValues = pd.MultiIndex.from_tuples(alignedTransformedOutlier.columns.values, names=['time', 'variables'])

#%% Analysis of online SPE
    
#Which variables account for most of the SPE rise

SPEVariables = pd.DataFrame(columns = columnsValues, index = badBatchesFoaming)

for batch in badBatchesFoaming:
    batchData = pd.DataFrame(alignedTransformedOutlier.loc[idx[batch],idx[:,:]]).T
    batchScaled = standardizeInput(batchData, mean_columnwise, std_columnwise, columnsValues)
    for k in range(len(alignedTransformedOutlier.columns.levels[0])):
        if k < (len(alignedTransformedOutlier.columns.levels[0])-1):
            batchScaledDev = copyDevVector(batchScaled, k)
            scoresBatch = calculateScores(batchScaledDev, prinComponents, prinComponents.index.values)
            projection = pd.DataFrame(np.dot(scoresBatch, prinComponents),
                                      index = batchData.index,
                                      columns = columnsValues)
            SPEVariables.loc[idx[batch], idx[k,:]] = ((batchData.loc[idx[:], idx[k,:]] - projection.loc[idx[:], idx[k,:]])**2).values
        else:
            SPEVariables.loc[idx[batch], idx[k,:]] = 0

#%%

def substract(x):
    return (x[1]-x[0])

diffSPETrans = (badBatchesSPE.T).rolling(center=False, window = 2).apply(func = substract)
diffSPE = diffSPETrans.T

#%%
x = diffSPE.idxmax(axis = 1).astype(int)
y = diffSPE.max(axis = 1)
SPEMax = pd.DataFrame([x.astype(int),y], index = ['time Index', 'slopeMax'])

#%%

p = SPEVariables.loc[idx[14],idx[0,:]].index.droplevel(0)

#%%
resultVar = {}

for batch in badBatchesFoaming:
    errorVar = SPEVariables.loc[idx[batch],idx[SPEMax.loc["time Index",batch],:]].values
    var = p[errorVar.argmax()]
    resultVar[batch] = var
    
resultVar
