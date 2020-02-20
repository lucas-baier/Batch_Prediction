# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:08:30 2017

@author: delubai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
from sklearn import preprocessing
idx = pd.IndexSlice

PYTHONHASHSEED = 0
random.seed(0)


#%%
def det_test_index(data, outlier, sizeTest):
    dataNormalAll = data[np.logical_not(data.index.isin(outlier))]
    sizeTestSet = int(round(sizeTest * len(dataNormalAll.index),0))
    testIndex = random.sample(list(dataNormalAll.index.values), sizeTestSet)
    return (testIndex)


#%%
def standardizeInput (data, mean, std, columnsValues):
    scaled = (data - mean)/(std)
#    scaled = pd.Series(((data.values - mean)/(std)), index = data.index)
#    scaled = pd.DataFrame(((data.loc[idx[:],idx[:,:]] - mean)/(std)), index = data.index,
#                                columns = columnsValues)                                
    return (scaled)
 
 
#%%
 
def standardize_data_current (data, dataTest, dataOutlier, columns, no_batches_scaling):
    dataScaled = pd.DataFrame(preprocessing.scale(data), 
                                            index = data.index, 
                                            columns = columns) 
    for batch in dataTest.index:
        selected_data = data[data.index < batch].iloc[-no_batches_scaling:,:]
        if (selected_data.shape[0] < no_batches_scaling):
            selected_data_new = data[data.index > batch].iloc[:(no_batches_scaling-selected_data.shape[0]),:]
            selected_data = pd.concat([selected_data, selected_data_new])
        mean = selected_data.mean(axis = 0)
        std = selected_data.std(axis = 0)
        std[std == 0] = 1
        dataTestScaled = (dataTest - mean)/(std)
        dataTestScaled.columns = columns
    
    for batch in dataOutlier.index:
        selected_data = data[data.index < batch].iloc[-no_batches_scaling:,:]
        if (selected_data.shape[0] < no_batches_scaling):
            selected_data_new = data[data.index > batch].iloc[:(no_batches_scaling-selected_data.shape[0]),:]
            selected_data = pd.concat([selected_data, selected_data_new])
        mean = selected_data.mean(axis = 0)
        std = selected_data.std(axis = 0)
        std[std == 0] = 1
        dataOutlierScaled = (dataOutlier - mean)/(std)
        dataOutlierScaled.columns = columns    
    
    return(dataScaled, dataTestScaled, dataOutlierScaled) 
#%%  
def calculate_online_SPE(data, principal_components, columnsValues, normal_data, no_batches_scaling):
    noVar = len(data.columns.levels[1])
    scoresColumns = pd.MultiIndex.from_product([data.columns.levels[0],principal_components.index], names = ['time', 'components'])
        
    SPE_df = pd.DataFrame()
    scoresBatches_df = pd.DataFrame(columns = scoresColumns)
    SPEVar_df = pd.DataFrame(columns = columnsValues)
    
    for batch in data.index:
        batchData = data.loc[batch]
        
        selected_data = normal_data[normal_data.index < batch].iloc[-no_batches_scaling:,:]
        if (selected_data.shape[0] < no_batches_scaling):
            selected_data_new = normal_data[normal_data.index > batch].iloc[:(no_batches_scaling-selected_data.shape[0]),:]
            selected_data = pd.concat([selected_data, selected_data_new])
        mean = selected_data.mean(axis = 0)
        std = selected_data.std(axis = 0)              
        std[std == 0] = 1#e-15        
        
        batchScaled = standardizeInput(batchData, mean, std, columnsValues)
        resultSPE = {}  
        listSPE = []
        listScores = []
        for k in range(len(data.columns.levels[0])):
            if k < (len(data.columns.levels[0])-1):
                batchScaledDev = copyDevVector(batchScaled, k)
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
def calculateSPE2 (scores, pComp, data, i, nVar):
    projection = np.dot(scores, pComp)
    SPEVar = ((data[(i*nVar):((i+1)*nVar)] - projection[(i*nVar):((i+1)*nVar)])**2)
    SPE = SPEVar.sum()
    return (SPE, SPEVar)

#%%
def copyDevVector (scaled, i):
    repeat = len(scaled.index.levels[0])-(i+1)
    currentDev = scaled[i]
    dev = np.tile(currentDev, repeat)
    knownData=scaled.loc[0:i].values
    devComplete = np.concatenate((knownData, dev))
    return (devComplete)
    
#%%   
def calculateScores (data, pComp, columnNamesPC):
    scores = pd.DataFrame(np.dot(data, pComp.T),
                          index = data.index,
                          columns = columnNamesPC)
    return (scores)
#%%   
def calculateSPE (scores, pComp, data, columnsValues, i):
    projection = pd.DataFrame(np.dot(scores, pComp),
                              index = data.index,
                              columns = columnsValues)
    SPE = ((data.loc[idx[:], idx[i,:]] - projection.loc[idx[:], idx[i,:]])**2).sum(axis = 1).values
    return (SPE)

#%%
def foamingPoint (data, batch):
    foamingTrajectory = data[data.batch == batch].loc[:,'LA20162.2'].reset_index(drop = True)
    foamingPoints = foamingTrajectory.index[foamingTrajectory > 0]
    return (foamingPoints[0])
    
#%%
    
def calcErrorMat (scores, pComp, data, columnsValues):
    projection = pd.DataFrame(np.dot(scores, pComp),
                              index = data.index,
                              columns = columnsValues)
    ErrorMatrix = (data - projection)
    return (ErrorMatrix)

#%%
def det_classification(SPEdata, SPEControl, alignedFoamingSignal = None):

    classification = {}
    for i in SPEdata.index.values:  
        if (alignedFoamingSignal is not None):
            foamP = foamingPoint(alignedFoamingSignal, i)
            binTable = (SPEdata[SPEdata.index == i].iloc[0,0:foamP-1] > SPEControl[0:foamP-1])
        else:
            binTable = (SPEdata[SPEdata.index == i].iloc[0,:] > SPEControl[0:])
        
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
  
def buildConfusionMatrix (classificationBadBatches, classificationGoodBatches):
    s1 = pd.Series([classificationBadBatches[0], classificationBadBatches[1]], index=['NoFault', 'Fault'], name='Fault')
    s2 = pd.Series([classificationGoodBatches[0], classificationGoodBatches[1]], index=['NoFault', 'Fault'], name='NoFault')
    confusionMatrix = pd.concat([s1,s2], axis = 1).T
    
    indexConfusion = pd.MultiIndex.from_tuples( [('Actual','Fault'),('Actual', 'NoFault')])
    columnsConfusion = pd.MultiIndex.from_tuples( [('Predicted','Fault'),('Predicted', 'NoFault')])
    confusionMatrix.index = indexConfusion
    confusionMatrix.columns = columnsConfusion
    return (pd.DataFrame(confusionMatrix))
    
#%%
    
def calcRates (classificationBadBatches, classificationGoodBatches):
    TPR = classificationBadBatches[0] / (classificationBadBatches[0] + classificationBadBatches[1])
    TNR = classificationGoodBatches[1] / (classificationGoodBatches[0] + classificationGoodBatches[1])
    FPR = classificationGoodBatches[0] / (classificationGoodBatches[0] + classificationGoodBatches[1])
    if ((classificationBadBatches[0] + classificationGoodBatches[0]) != 0):    
        Precision = classificationBadBatches[0] / (classificationBadBatches[0] + classificationGoodBatches[0])
    else:
        Precision = 'NaN'
    result = pd.Series([TPR, TNR, FPR, Precision], index = ['TPR', 'TNR','FPR', 'Precision'])
    return (result)

#%%
   
def return_output_ROC(confidence_matrix, kpi, alpha):
    TP = confidence_matrix.iloc[0,0]
    FN = confidence_matrix.iloc[0,1]
    FP = confidence_matrix.iloc[1,0]
    TN = confidence_matrix.iloc[1,1]    
    TPR = kpi.loc['TPR']
    FPR = kpi.loc['FPR']
    result = pd.Series([TP, FN, FP, TN, TPR, FPR, alpha], 
                       index = ['TP','FN','FP','TN','TPR','FPR', 'alpha'])
    return (result)
#%%
    
def barplotVarSPE (batchesSPEVar, batch, t):
    QTime = batchesSPEVar[batchesSPEVar.index == batch].stack(0)
    valuesPlot = QTime[QTime.index.levels[1] == t]
    x = range(valuesPlot.shape[1])
    y = valuesPlot.values[0].tolist()
    labels = (valuesPlot.columns.values).tolist()
    
    fig, ax = plt.subplots()
    
    _ = plt.bar(x, y, align='center', color = 'b')
    _ = plt.xticks(x, labels)
    #_ = ax.set_ylim([0,500])
    
    _ = plt.xlabel('Variables')
    _ = plt.ylabel('Q')
    _ = plt.title('Overview on prediction error per variable, batch %i, t = %i' %(batch, t))
    _ = plt.legend()
    
    _ = plt.show();
    
#%%
    
def barplot_Q_model (batches_Q_per_variable, batch):
    data = batches_Q_per_variable[batches_Q_per_variable.index == batch]
    x = range(data.shape[1])
    y = data.values[0].tolist()
    labels = (data.columns.values).tolist()
    
    fig, ax = plt.subplots()
    
    _ = plt.bar(x, y, align='center', color = 'b')
    _ = plt.xticks(x, labels)
    #_ = ax.set_ylim([0,500])
    
    _ = plt.xlabel('Variables')
    _ = plt.ylabel('Q')
    _ = plt.title('Overview on prediction error per variable, batch %i' %(batch))
    _ = plt.legend()
    
    _ = plt.show();
    
    return(fig)
    
#%%
    
def abnormalBatchesIndex (data, foaming_signal):
    
# Removal of abnormal batches
    
    # Batches for the fault scenarios with foaming signal
    scenarioSteam = [90,105,107,133,165,166,171,185,193,240,256,257,259,260,
                     261,268,284]
    scenarioPostSteam = [14,27,40,67,92,100,102,103,110,130,132]
    scenarioOutlier = [26,244]
    
    scenario_bad_batches_new_data = [256, 257, 259, 260, 261, 268, 284]
    
    all_foaming_batches = list(foaming_signal[foaming_signal['LA20162.2'] == 1].batch.unique())
    # Batches for the fault scenarios with abnormal steam input behaviour
    scenarioSteamReduction = [2,6,15,41,77,79,83,86,87,89,94,96,101,111,113,118,136,156,158,174,212,223,229,230,232,235]
    
    # Batches which can be classified as outlier batches
    outlierStirrerPower = [37,74,98,99,124,125,126,127,149,163,164,167,168,169,170,174,181,182,183,184,194,
                           200,201,202,203,213,214,231,236,237,239,245,246]#, 255, 262, 265, 267]
    outlierBatchDuration = [13,32,34,44,57,58,73,97,123,162,176,182]
    
    batchesDestillat = [47,139,177,228]    
    
    unusualBatches = [8, 10, 20, 33, 48, 65, 72, 78, 82, 91, 134, 205, 215, 220, 221,233]
    unusualBatches2 = [71, 122, 131, 142, 143, 222, 242, 243, 247]
    #unusual_batches_new = [222,224,225,226,227,234,238, 241,242]
    
    
    unusual_batches_new_data = [283, 290, 295, 250,251]   
    
    allOutliers = list(set(all_foaming_batches + scenarioOutlier + 
                           scenarioSteamReduction + outlierStirrerPower + 
                           outlierBatchDuration + unusualBatches + unusualBatches2 +
                           batchesDestillat + unusual_batches_new_data))
                           
    badBatchesFoaming = data.index[data.index.isin(list(all_foaming_batches))]
    
    return (allOutliers, badBatchesFoaming)
    
#%%
def Q_contribution_plot_values(residual_matrix_squared):
    indexVar = pd.MultiIndex.from_tuples(residual_matrix_squared.columns.values, names=['time', 'variables'])
    residual_matrix_squared = pd.DataFrame(residual_matrix_squared, columns = indexVar)
    sum_var = residual_matrix_squared.groupby(axis = 1, level = 'variables').sum()
    return(sum_var)
    
#%%
def SPE_per_time_interval (residual_matrix, batch, var):
    x = (pd.Series(residual_matrix[residual_matrix.index == batch]
                    .loc[idx[:],idx[:,var]].values[0]))
    d = {}
    for i in [100,200,300,400,500,600]:
        d[i] = x[(x.index < i) & (x.index > (i-100))].sum()
    
    return(pd.Series(d))

#%%

def plot_trajectories_and_projection (data, batch, scores, var, principal_components, columns_values):
    time_series = (pd.Series(data[data.index == batch]
                    .loc[idx[:],idx[:,var]].values[0])) 
                    
    time_series_projection = pd.DataFrame(np.dot(scores[scores.index == batch], principal_components),
                                          index = [batch],
                                          columns = columns_values)
    time_series_var_projection = (pd.Series(time_series_projection.loc[idx[:],idx[:,var]].values[0]))                                       
    
    ax, fig = plt.subplots()
    _ = plt.plot(time_series, label = 'original data', color = 'blue')
    _ = plt.plot(time_series_var_projection, label = 'projection', color = 'red')
    _ = plt.legend()
    return (fig)
    
#%%
    
def plot_scatterplot_scores(scores, scores_foaming, scores_test, xAxis, yAxis, label = None):
    fig, ax = plt.subplots()
        
    _ = plt.scatter(scores.loc[:,xAxis], 
                scores.loc[:,yAxis],
                color = 'black',
                label = 'normal Batches')
    
    _ = plt.scatter(scores_foaming.loc[:,xAxis], 
                scores_foaming.loc[:,yAxis],
                color = 'r',
                label = 'bad Batches')
                
    _ = plt.scatter(scores_test.loc[:,xAxis], 
                scores_test.loc[:,yAxis],
                color = 'orange',
                label = 'test Batches')         
    
    if (label is not None):    
        for i, txt in enumerate(label.index):
            plt.annotate(txt, (label.loc[txt].loc[xAxis],label.loc[txt].loc[yAxis]))
    
    _ = fig.suptitle('Scatterplot of first two Principal Components', fontsize = 16)
    _ = plt.xlabel(xAxis)
    _ = plt.ylabel(yAxis)
    
    _ = plt.legend()
    
    _ = plt.show();
    
#%%
    
def plot_barchart_Q(Q_normal, Q_foaming, Q_test, Q_alpha):

    fig, ax = plt.subplots()
    
    #bar_width = 0.35
    
    _ = plt.bar(Q_normal.index, 
                Q_normal.iloc[:,0], 
                color = 'lightsteelblue', 
                label = 'Training set')
    
    _ = plt.bar(Q_foaming.index, 
                Q_foaming.iloc[:,0], 
                color = 'orangered', 
                label = 'Test set - foaming batches')
            
    _ = plt.bar(Q_test.index, 
                Q_test.iloc[:,0], 
                color = 'darkblue', 
                label = 'Test set - normal batches')
    
    ax.axhline(Q_alpha[0], label = '99% - confidence limit', color = 'black') #'-',
    #ax.axhline(Q_alpha[1], label = '95% - confidence limit', color = 'black')  #'--',
    
    ax.set_ylim([0,10000])
    ax.set_xlim([0,150])
#    ax.set_xlim([220,250])
    
    plt.xlabel('Batches')
    plt.ylabel('Q')
    #plt.title('Overview on sum of Squares of Residuals')
    plt.legend(prop={'size':20})
    
    plt.show();
    return(fig)
 
#%%
   
def det_test_foaming_batches_new(data, foaming_signal, var):
    data = data.loc[:,idx[:,var]]
    columnsValues = pd.MultiIndex.from_tuples(data.columns.values, names=['time', 'variables'])    
    
    data = data[data.index > 247]    
    foaming_batches = foaming_signal[foaming_signal['LA20162.2'] > 0].loc[:,'batch'].unique()
    batches_foaming = data[data.index.isin(foaming_batches)]
    batches_foaming.colums = columnsValues
    batches_test = data[np.logical_not(data.index.isin(foaming_batches))]
    batches_test.columns = columnsValues
    
    return(batches_test, batches_foaming)
    
#%%
def det_normal_test_foaming_batches_new(data, foaming_signal, var, test_index):
    data = data.loc[:,idx[:,var]]
    columnsValues = pd.MultiIndex.from_tuples(data.columns.values, names=['time', 'variables'])    
    
    data = data[data.index > 247]    
    foaming_batches = foaming_signal[foaming_signal['LA20162.2'] > 0].loc[:,'batch'].unique()
    
    batches_foaming = data[data.index.isin(foaming_batches)]
    batches_foaming.colums = columnsValues
    batches_normal = data[(np.logical_not(data.index.isin(foaming_batches))) & (np.logical_not(data.index.isin(test_index)))]
    batches_normal.columns = columnsValues
    batches_test = data[data.index.isin(test_index)]
    batches_test.columns = columnsValues
    return(batches_normal, batches_test, batches_foaming)

#%%
def det_test_index_new_data(data, foaming_batches):
    new_outliers = [283, 290, 295, 250,251]    
    index_list = list((set(list(data[data.index > 247].index)) - set(list(foaming_batches))) - set(new_outliers))
    
    # - set(foaming_batches)# - set(new_outliers)
    
    size = int(np.round(len(index_list)*0.3, 0))
    test_index = random.sample(index_list, size)
    return(test_index)
    
#%%

def calc_output_accuracy(SPE_confidence_matrix):
    SPE_values = pd.Series([SPE_confidence_matrix.iloc[0,0],SPE_confidence_matrix.iloc[0,1],
                            SPE_confidence_matrix.iloc[1,0],SPE_confidence_matrix.iloc[1,1]], 
                            index = ['TP_Q','FN_Q','FP_Q','TN_Q'])
    Accuracy = (SPE_values[0] + SPE_values[3]) / (SPE_values.sum())
    SPE_values['Accuracy'] = Accuracy
    return (SPE_values)

    
#%%
    
test_index_old_batches = [210,
 85,
 192,
 219,
 104,
 11,
 54,
 135,
 119,
 93,
 197,
 208,
 61,
 117,
 70,
 148,
 46,
 129,
 29,
 59,
 199,
 191,
 22,
 154,
 53,
 140,
 180,
 152,
 30,
 62,
 195,
 18,
 175,
 66,
 116,
 145]

# determined using function det_test_index_new_data
 
test_index_new_batches = [249, 293, 320, 296, 264, 280, 302, 301, 294, 285, 300,
                          289, 308, 277, 322, 272, 282]
                          
scenarioSteam = [90,105,107,133,165,166,171,185,193,240,256,257,259,260,
                     261,268,284]
scenarioPostSteam = [14,27,40,67,92,100,102,103,110,130,132]

#%%

# Hard code the batch numbers for test & training batches, indexes are chosen randomly. Fix it since
# random changes its output every time. 

index_training_batches = [  1,   3,   4,   5,   9,  12,  16,  17,  19,  21,  23,  24,  25,
                            28,  31,  35,  36,  38,  39,  42,  43,  45,  49,  50,  51,  52,
                            55,  56,  60,  63,  64,  68,  69,  80,  81,  84,  88,  95, 106,
                           108, 109, 112, 114, 115, 128, 137, 138, 141, 144, 146, 147, 150,
                           151, 153, 155, 157, 159, 160, 161, 172, 173, 178, 179, 186, 187,
                           188, 189, 190, 196, 198, 204, 206, 207, 209, 211, 216, 217, 218,
                           224, 225, 226, 227, 234, 238, 241, 248, 252, 255, 262, 263, 265,
                           267, 269, 270, 271, 273, 274, 275, 276, 278, 279, 281, 287, 288,
                           291, 297, 298, 299, 303, 304, 305, 306, 309, 310, 311, 313, 314,
                           315, 316, 317, 318, 319, 321, 323, 324]
                           
index_test_batches = [ 11,  18,  22,  29,  30,  46,  53,  54,  59,  61,  62,  66,  70,
                        85,  93, 104, 116, 117, 119, 129, 135, 140, 145, 148, 152, 154,
                       175, 180, 191, 192, 195, 197, 199, 208, 210, 219, 249, 264, 272,
                       277, 280, 282, 285, 289, 293, 294, 296, 300, 301, 302, 308, 320, 322]
                       
                       
index_foaming_batches = [ 14,  27,  67,  90,  92, 100, 102, 103, 105, 107, 133, 165, 166,
                         171, 185, 193, 256, 257, 259, 260, 261, 268, 284]

#%%

index_validation_foaming = [100, 14, 193, 171, 259, 256, 107, 165, 102, 260, 133]
index_test_validation_foaming = [257, 67, 261, 166, 103, 105, 268, 92, 185, 90, 27, 284]

index_validation = [145, 180,  66, 29, 199, 192, 22, 272, 53, 154, 70, 116, 322,
                    195, 249, 62, 85, 302, 264, 210, 140, 282, 219, 129, 93, 301]
index_test_validation = [135,  11,  18, 148, 277, 280, 152, 285,  30, 289, 293,
                         294, 296, 300,  46, 175, 308,  54,  59,  61, 191, 320,
                         197, 208, 104, 117, 119]