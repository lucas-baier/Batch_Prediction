# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:39:02 2017

@author: delubai
"""

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random 
idx = pd.IndexSlice

PYTHONHASHSEED = 0
random.seed(0)


#%%
def det_eigvectors(kernel_matrix, no_components):
# Get the eigenvector with largest eigenvalue
    eigvals, eigvecs = sp.linalg.eigh(kernel_matrix)
    
    indices = eigvals.argsort()[::-1]  
    eigvals = eigvals[indices]
    eigvecs = eigvecs[:, indices]
    
    eigvals_full = eigvals[eigvals > 0]
    eigvecs_full = eigvecs[:, eigvals > 0]/np.sqrt(eigvals_full)   
    
    eig_values = eigvals_full[:no_components]     
    eig_vectors = eigvecs_full[:,:no_components]

    return(eig_vectors, eig_values, eigvecs_full, eigvals_full)
    
#%%
    
def calculate_kernel_matrix_rbf(dataScaled, gamma):
    dist = sp.spatial.distance.pdist(dataScaled, 'sqeuclidean')
    square_matrix_dist = sp.spatial.distance.squareform(dist)
    kernel_matrix = np.exp(-gamma * square_matrix_dist)
    
    kern_centerer = preprocessing.KernelCenterer()
    center_model = kern_centerer.fit(kernel_matrix)
    kernel_matrix_centered = center_model.transform(kernel_matrix)
    #kernel_matrix_scaled = (kernel_matrix_centered) / (np.matrix.trace(kernel_matrix_centered)/(len(dataScaled.index)-1))
    return (pd.DataFrame(kernel_matrix_centered, index = dataScaled.index,
                         columns = dataScaled.index), center_model)
#%%
def other_kernel_matrix_rbf_offline(normal_batches, other_batches, center_model, gamma):
    if (isinstance(other_batches, pd.core.series.Series)):
        index_result = other_batches.name
    if (isinstance(other_batches, pd.core.frame.DataFrame)):
        index_result = other_batches.index
    kernel_matrix = sk.metrics.pairwise.rbf_kernel(other_batches, normal_batches, 
                                                   gamma = gamma)
    kernel_matrix = center_model.transform(kernel_matrix)
    return (pd.DataFrame(kernel_matrix, index = [index_result],
                         columns = normal_batches.index))

#%%
def other_kernel_matrix_rbf(normal_batches, other_batches, center_model, gamma):
    if (isinstance(other_batches, pd.core.series.Series)):
        index_result = other_batches.name
    if (isinstance(other_batches, pd.core.frame.DataFrame)):
        index_result = other_batches.index
    kernel_matrix = sk.metrics.pairwise.rbf_kernel(other_batches.reshape(1,-1),
                                                   normal_batches, 
                                                   gamma = gamma)
    kernel_matrix = center_model.transform(kernel_matrix)
    return (pd.DataFrame(kernel_matrix, index = [index_result],
                         columns = normal_batches.index))
                         
#%%
def calculate_kernel_matrix_poly(dataScaled, degree, coef0):
    kernel_matrix = sk.metrics.pairwise.polynomial_kernel(dataScaled, degree = degree, coef0 = coef0)
    
    kern_centerer = preprocessing.KernelCenterer()
    center_model = kern_centerer.fit(kernel_matrix)
    kernel_matrix_centered = center_model.transform(kernel_matrix)
    #kernel_matrix_scaled = (kernel_matrix_centered) / (np.matrix.trace(kernel_matrix_centered)/(len(dataScaled.index)-1))    
    return (pd.DataFrame(kernel_matrix_centered, index = dataScaled.index,
                         columns = dataScaled.index), center_model)
    
#%%
def other_kernel_matrix_poly(normal_batches, other_batches, center_model, degree, coef0):
    if (isinstance(other_batches, pd.core.series.Series)):
        index_result = other_batches.name
    if (isinstance(other_batches, pd.core.frame.DataFrame)):
        index_result = other_batches.index
    kernel_matrix = sk.metrics.pairwise.polynomial_kernel(other_batches.reshape(1,-1), 
                                                          normal_batches, degree = degree, coef0 = coef0)
    kernel_matrix = center_model.transform(kernel_matrix)
    return (pd.DataFrame(kernel_matrix, index = [index_result],
                         columns = normal_batches.index))

#%%
def other_kernel_matrix_poly_offline(normal_batches, other_batches, center_model, degree, coef0):
    if (isinstance(other_batches, pd.core.series.Series)):
        index_result = other_batches.name
    if (isinstance(other_batches, pd.core.frame.DataFrame)):
        index_result = other_batches.index
    kernel_matrix = sk.metrics.pairwise.polynomial_kernel(other_batches, 
                                                          normal_batches, degree = degree, coef0 = coef0)
    kernel_matrix = center_model.transform(kernel_matrix)
    return (pd.DataFrame(kernel_matrix, index = [index_result],
                         columns = normal_batches.index))


 #%%
def other_kernel_matrix(normal_batches, other_batches, center_model, gamma):
    merged_batches = pd.concat([normal_batches, other_batches])
    dist = sp.spatial.distance.pdist(merged_batches, 'sqeuclidean')
    square_matrix_dist = pd.DataFrame(sp.spatial.distance.squareform(dist),
                                      index = merged_batches.index,
                                      columns = merged_batches.index)
    square_reduced = square_matrix_dist[(square_matrix_dist.index.isin(other_batches.index))]
    square_reduced = square_reduced.iloc[:,:len(normal_batches.index)]
    kernel_matrix_outlier = np.exp(-gamma * square_reduced)
    kernel_matrix_outlier = center_model.transform(kernel_matrix_outlier)  
    return (pd.DataFrame(kernel_matrix_outlier, index = other_batches.index,
                         columns = normal_batches.index))   
                         
#%%
def calc_scores (kernel_matrix, alphas):
    columnnamesPC = ["PC%i" %(s) for s in range(1,alphas.shape[1] + 1)]
    scores = pd.DataFrame(np.dot(kernel_matrix, alphas), columns = columnnamesPC,
                          index = kernel_matrix.index)
    return (scores)
    
#%%
    
def det_test_index(testIndex):
    index_validation_test = random.sample(list(testIndex), int(round(len(testIndex)*0.5, 0)))
    return (index_validation_test)