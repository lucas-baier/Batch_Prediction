# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:30:30 2017

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
import sklearn as sk
import MPCA_functions 
import modular_MPCA_main
import KPCA_functions
import importlib

importlib.reload(MPCA_functions)
importlib.reload(KPCA_functions)
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

# Batches for the fault scenarios with foaming signal
scenarioSteam = [90,105,107,133,165,166,171,185,193,240,256,257,259,260,261,
                 268,284]
scenarioPostSteam = [14,27,40,67,92,100,102,103,110,130,132]
scenarioOutlier = [26,244] 

pathToDataSets="C://project-data/my-data-sets/data_new_processed/"
pathToOutput = "C://project-data/my-data-sets/Kernel_MPCA/"

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

#%%
def split_data (var, data, index_training, index_test, index_foaming):
    data = data.loc[:,idx[:,var]]
    columnsValues = pd.MultiIndex.from_tuples(data.columns.values, names=['time', 'variables'])
    
    dataNormal = data[data.index.isin(index_training)] 
    dataNormal.columns = columnsValues
    
    dataTest = data[data.index.isin(index_test)]
    dataTest.columns = columnsValues
    
    dataOutlier = data[data.index.isin(index_foaming)]
    dataOutlier.columns = columnsValues
    
    return(dataNormal, dataTest, dataOutlier, columnsValues)

#%%

batchesNormal, batchesTest, batchesOutlier, columnsValues = split_data(
    varArray, alignedTransformed, index_training, index_test, index_foaming)

gamma = 1/(3.5*batchesNormal.shape[1])

batchesNormalScaled, batchesTestScaled, batchesOutlierScaled, mean, std = modular_MPCA_main.standardize_data(
    batchesNormal, batchesTest, batchesOutlier, columnsValues)


#%%
kernel_matrix_poly, kernel_centerer_poly = KPCA_functions.calculate_kernel_matrix_poly(batchesNormalScaled, degree_poly, coef0_poly)
     
kernel_matrix_rbf, kernel_centerer_rbf = KPCA_functions.calculate_kernel_matrix_rbf(batchesNormalScaled, gamma)    

#%%

#alphas, lambdas, alphas_full, lambdas_full = det_eigvectors(kernel_matrix, no_components)
alphas_poly, lambdas_poly, alphas_full_poly, lambdas_full_poly = KPCA_functions.det_eigvectors(kernel_matrix_poly, no_components)

alphas_rbf, lambdas_rbf, alphas_full_rbf, lambdas_full_rbf = KPCA_functions.det_eigvectors(kernel_matrix_rbf, no_components)


#%%
    
kernel_matrix_outlier_poly = KPCA_functions.other_kernel_matrix_poly_offline(batchesNormalScaled, batchesOutlierScaled,
                                                      kernel_centerer_poly, degree_poly, coef0_poly)
    
kernel_matrix_test_poly = KPCA_functions.other_kernel_matrix_poly_offline(batchesNormalScaled, batchesTestScaled,
                                                      kernel_centerer_poly, degree_poly, coef0_poly)

#%%

kernel_matrix_outlier_rbf = KPCA_functions.other_kernel_matrix_rbf_offline(batchesNormalScaled,
                                                                           batchesOutlierScaled,
                                                                           kernel_centerer_rbf,
                                                                           gamma)

kernel_matrix_test_rbf = KPCA_functions.other_kernel_matrix_rbf_offline(batchesNormalScaled,
                                                                        batchesTestScaled,
                                                                       kernel_centerer_rbf,
                                                                       gamma)

#%%
# Calculate scoes based on poly kernel

scores_normal = KPCA_functions.calc_scores(kernel_matrix_poly, alphas_poly)
scores_outlier = KPCA_functions.calc_scores(kernel_matrix_outlier_poly, alphas_poly)
scores_test = KPCA_functions.calc_scores(kernel_matrix_test_poly, alphas_poly)

scores_full_normal = KPCA_functions.calc_scores(kernel_matrix_poly, alphas_full_poly)
scores_full_outlier = KPCA_functions.calc_scores(kernel_matrix_outlier_poly, alphas_full_poly)
scores_full_test = KPCA_functions.calc_scores(kernel_matrix_test_poly, alphas_full_poly)

scores_steam = scores_outlier[scores_outlier.index.isin(scenarioSteam)]
scores_poststeam = scores_outlier[scores_outlier.index.isin(scenarioPostSteam)]

#%%

# Calculate scores based on rbf kernel

scores_normal = KPCA_functions.calc_scores(kernel_matrix_rbf, alphas_rbf)
scores_outlier = KPCA_functions.calc_scores(kernel_matrix_outlier_rbf, alphas_rbf)
scores_test = KPCA_functions.calc_scores(kernel_matrix_test_rbf, alphas_rbf)

scores_full_normal = KPCA_functions.calc_scores(kernel_matrix_rbf, alphas_full_rbf)
scores_full_outlier = KPCA_functions.calc_scores(kernel_matrix_outlier_rbf, alphas_full_rbf)
scores_full_test = KPCA_functions.calc_scores(kernel_matrix_test_rbf, alphas_full_rbf)

#%%
fig, ax = plt.subplots()

xAxis = 'PC1'
yAxis = 'PC2'
    
plt.scatter(scores_normal[xAxis],
            scores_normal[yAxis],
            color = 'b',
            label = 'normal Batches')
            
plt.scatter(scores_outlier[xAxis],
            scores_outlier[yAxis],
            color = 'r',
            label = 'bad Batches')
            
plt.scatter(scores_test[xAxis],
            scores_test[yAxis],
            color = 'orange',
            label = 'test Batches')

#plt.scatter(scores_test[xAxis],
#            scores_test[yAxis],
#            color = 'g',
#            label = 'test Batches')
   
 
#for i, txt in enumerate(scores_normal.index):
#    plt.annotate(txt, xy = (scores_normal.iloc[i,0],scores_normal.iloc[i,1]))

#ax.set_xlim([-1,1])
#ax.set_ylim([-1,1])
fig.suptitle('Scatterplot of first two Principal Components', fontsize = 16)
plt.xlabel(xAxis)
plt.ylabel(yAxis)
#plt.legend()
plt.show();

#%%


covMatrix = np.cov(scores_normal.T)
covMatrix[covMatrix < 0.00001] = 0
invCovMatrix = np.linalg.inv(covMatrix)

alphaD = 0.99
#%% Calculate hotelling statistic
#invEigvalues = np.linalg.inv(np.diag(lambdas_poly))

def calculate_T_stat (scores):
    result = np.dot(np.dot(scores, invCovMatrix), scores.T)
    return (result)

T_squared = scores_normal.apply(calculate_T_stat, axis = 1)
T_squared_outlier = scores_outlier.apply(calculate_T_stat, axis = 1)
T_squared_test = scores_test.apply(calculate_T_stat, axis = 1)

controlLimit = (((no_components * ((len(scores_normal.index)**2)-1))/
                (len(scores_normal.index)* (len(scores_normal.index) - no_components))) * 
                sp.stats.f.ppf(alphaD, no_components, (len(scores_normal.index) - no_components)))

#%% Plot t statistic and control limits

fig, ax = plt.subplots()

plt.bar(T_squared.index,
        T_squared.values)
        
plt.bar(T_squared_outlier.index,
        T_squared_outlier.values,
        color = 'r')
        
plt.bar(T_squared_test.index,
        T_squared_test.values,
        color = 'g')
ax.axhline(controlLimit, label = '95% - confidence limit', color = 'black')
plt.show();

#%%

def calculate_SPE (scores, scores_full):
    SPE = (scores_full**2).sum(axis = 1) - (scores**2).sum(axis = 1)
    return (SPE)

#%%
SPE_normal = calculate_SPE(scores_normal, scores_full_normal)
SPE_outlier = calculate_SPE(scores_outlier, scores_full_outlier)
SPE_test = calculate_SPE(scores_test, scores_full_test)

#%% Caluclate confidence limit

def calculate_Q_conf_limit(SPE_normal, alpha):
    variance = SPE_normal.var()
    mean = SPE_normal.mean()
    g = variance / (2*mean)
    h = round(((2*(mean**2))/(variance)),0)
    conf_limit_SPE = g * sp.stats.chi2.ppf(alpha, h) 
    return (conf_limit_SPE)
#%%
    
SPE_conf_limit = calculate_Q_conf_limit(SPE_normal, alphaQ)
#%%
fig, ax = plt.subplots()

plt.bar(SPE_normal.index, 
        SPE_normal.values, 
        color = 'b', 
        label = 'Good batches')

plt.bar(SPE_outlier.index, 
        SPE_outlier.values, 
        color = 'r', 
        label = 'Foaming batches')
        
plt.bar(SPE_test.index, 
        SPE_test.values, 
        color = 'g', 
        label = 'Test batches')

ax.axhline(SPE_conf_limit, label = '99% - confidence limit', color = 'black')

#ax.set_ylim([0,100])
plt.xlabel('Batches')
plt.ylabel('Q')
plt.title('Overview on Sum of Squares of Residuals')
plt.legend()

plt.show();