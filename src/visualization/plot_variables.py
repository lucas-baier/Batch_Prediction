# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:36:45 2017

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
#from sklearn import preprocessing
#from sklearn import decomposition
#import MPCA_functions 
#import modular_MPCA_main

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
pathToDataSets="C://project-data/my-data-sets/merged/"
pathToDataSets_new="C://project-data/my-data-sets/data_new_processed/"
pathToOutput="C://project-data/my-data-sets/test/"

#data_set = pd.read_pickle(os.path.join(pathToDataSets,'processed.pkl'))
#batch_timing = pd.read_pickle(os.path.join(pathToDataSets, 'batch_timing.pkl'))
#phase_transition_points = pd.read_csv(os.path.join(pathToDataSets, 'phaseTransitionPoints.csv'))

data_set_with_alignment = pd.read_csv(os.path.join(pathToDataSets_new,'aligned_batches_all_var_all.csv'))
batch_timing_new = pd.read_csv(os.path.join(pathToDataSets_new,'batch_timing_all.csv'))
aligned_foaming_signal = pd.read_csv(os.path.join(pathToDataSets_new,'alignedFoamingSignal_scaled_var_all.csv'))
data_set_without_alignment = pd.read_csv(os.path.join(pathToDataSets_new,'processed_scaled_all_var.csv'))
#Show plots, 1 = yes, 0 = no
plots_on = 1


#batch_timing['batch'] = range(1,83)
#batch_timing.to_pickle(os.path.join(pathToDataSets_new,'batch_timing.pkl'))

#%%
# some batches are labelled as golden
#
golden_batches = [59,60,108]
reference_batch = 108

bad_batches_foaming = [ 14,  27,  67,  90,  92, 100, 102, 103, 105, 107, 133,
                       165, 166, 171, 185, 193, 256, 257, 259, 260, 261, 268, 284]
                       
bad_batches_foaming_steam = [90,105,107,133,165,166,171,185,193,240]

batches_unusual_steam = [2,6,15,41,77,79,83,86,87]
                         #89,94,96,101,111,113,118,136,156,158,174,212,223,229,230,232,235

foaming_batches_new = [261, 262, 264, 265, 266, 273, 289]

unusual_steam_behaviour_unwarped_1 = [105, 212, 102, 79, 229, 235, 165, 214, 162, 163, 183, 236,
                                      125, 231, 167, 168, 74, 164, 181, 289, 270, 269, 222, 233, 
                                      273, 275, 131, 298, 174, 149, 245, 239, 282, 202, 203, 312,
                                      284, 267, 296, 265, 194, 200, 47, 139, 307, 256, 97, 130, 171,
                                      123, 182, 290]


unusual_steam_behaviour_unwarped_2 = [78, 90, 166, 58, 124, 221, 266, 283, 76,
                                      77, 48, 111, 158, 230, 41, 136, 156, 113,
                                      87, 89, 118, 2, 6, 94, 83, 101, 242, 277,
                                      96, 232, 186, 223, 15, 86]
   

unusual_steam_test = [73, 195, 207, 199, 302, 206, 224, 115, 151, 274, 295, 316, 286, 303,
                      306, 299, 314, 228, 161, 300, 263, 311, 226, 287]
                          
all_batches = list(data_set_with_alignment.batch.unique())

unusual_steam_behaviour_1 = [i for i in unusual_steam_behaviour_unwarped_1 if i in all_batches]
unusual_steam_behaviour_2 = [i for i in unusual_steam_behaviour_unwarped_2 if i in all_batches]
unusual_steam_test = [i for i in unusual_steam_test if i in all_batches]



unusual_stirrer_behaviour = [37, 262, 133, 259, 283, 284, 127, 166, 193, 268, 163, 167, 246, 261,
                             239, 255, 256, 245, 267, 74, 149, 182, 183, 185, 174, 181, 200, 124,
                             165, 90, 98, 264, 260, 265, 164, 213, 214, 296, 312, 201, 231, 237, 
                             202, 203, 236, 282, 170, 194, 184, 125, 168, 126, 169]
  
unusual_stirrer_behaviour = [i for i in unusual_stirrer_behaviour if i in all_batches]
                           
test =  [106,  53,  61,  65, 140,  69, 314, 112, 119, 204] 

unusual_batches_new = [245, 275, 283, 290, 295]
#%%

def foaming_points (data, batch):
    foamingTrajectory = data[data.batch == batch].loc[:,'LA20162.2'].reset_index(drop = True)
    foamingPoints = foamingTrajectory.index[foamingTrajectory > 0]
    return (foamingPoints)
    
#%%
    
variable_plot1 ='E20161'
variable_plot2 = 'PC20162'
variable_plot3 ='FC20144'
#foamingSignal ="LA20162.2"

#%%
data_set_new = data_set_with_alignment
show_reference_batch = 1
show_phase_transition = 0


#%%
test_batches_false_prediction =  [254, 255, 260, 269, 294, 319]
batches_post_steam = [14,27,40,67,92,100,102,103,110,130,132]
new_batches_training_set = [248, 252, 255, 262, 263, 265, 267, 269, 270, 271,
                            273, 274, 275, 276, 278, 279, 281, 287, 288, 291, 
                            297, 298, 299, 303, 304, 305, 306, 309, 310, 311, 
                            313, 314, 315, 316, 317, 318, 319, 321, 323, 324]
 
for batch in [260]:
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)    

    fig, ax1 = plt.subplots()
    _ = fig.suptitle('Batch {} and Ref Batch {}'.format(batch, reference_batch))
    
    #x = trajectorySPE.columns.values
    trajectory_1 = (data_set_new[data_set_new.batch == batch].loc[:,variable_plot1]
                        .reset_index(drop =True))
    trajectory_reference_1 = (data_set_new[data_set_new.batch == reference_batch]
                            .loc[:,variable_plot1]
                            .reset_index(drop =True))
    
    ln1 = ax1.plot(trajectory_1, label = '{} Batch {}'.format(variable_plot1, batch),
                   marker = 'o',
                 linestyle='-', markersize = 3, color = 'red')
    _ = ax1.set_xlabel('time (min)')
    _ = ax1.set_ylabel(variable_plot1)
    
    if (batch in bad_batches_foaming):
        foaming_points_batch = foaming_points(aligned_foaming_signal, batch)
        for i in foaming_points_batch:
            _ = ax1.axvline(i, color = 'blue', label = 'First foaming signal')

    
    if (show_reference_batch == 1):     
        ln2 = ax1.plot(trajectory_reference_1, label = '{} Ref Batch {}'.format(variable_plot1, reference_batch),
                     marker = 'o',
                     linestyle='-', markersize = 3, color = 'black')  
                 
    
    if (show_phase_transition == 1):
        points = phase_transition_points[phase_transition_points.index == batch]        
        for i in points.iloc[0,:].values:
            _ = ax1.axvline(i, color = 'red', label = 'Phase transition points')
        
    ax2 = ax1.twinx()
    trajectory_2 = (data_set_new[data_set_new.batch == batch].loc[:,variable_plot2]
                        .reset_index(drop = True))
    trajectory_reference_2 = (data_set_new[data_set_new.batch == reference_batch]
                                .loc[:,variable_plot2]
                                .reset_index(drop = True))                        
    ln3 = ax2.plot(trajectory_2, label = '{} Batch {}'.format(variable_plot2, batch),
                 color = 'red')
                 
    if (show_reference_batch == 1): 
        ln4 = ax2.plot(trajectory_reference_2, label = '{} Ref Batch {}'.format(variable_plot2, reference_batch),
                     color = 'black')
                     
    _ = ax2.set_ylabel(variable_plot2)
    #_ = ax2.set_ylim([0, 2*max(trajectory_2)])
    
    trajectory_3 = (data_set_new[data_set_new.batch == batch].loc[:,variable_plot3]
                        .reset_index(drop = True))
    trajectory_reference_3 = (data_set_new[data_set_new.batch == reference_batch]
                                .loc[:,variable_plot3]
                                .reset_index(drop = True))
                                
    #ax3 = ax2.twinx()
    ln5 = ax2.plot(trajectory_3, label = '{} Batch {}'.format(variable_plot3, batch),
                 color = 'orange')
    if (show_reference_batch ==1): 
        ln6 = ax2.plot(trajectory_reference_3, label = '{} Ref Batch {}'.format(variable_plot3, reference_batch),
                     color = 'green')
    #_ = ax1.set_ylim(0,500)
    #_ = ax1.set_xlim(210,235)
    #_ = ax1.axvline(foamingPoints[0], color = 'red', label = 'First foaming signal')
    
    lns = ln1+ln2+ln3+ln4+ln5+ln6
    labs = [l.get_label() for l in lns]
    _ = ax1.legend(lns, labs, loc=1)    
    
    plt.show()

#%%
#reference_batch = 108
#data_set_without_alignment = data_set_with_alignemnt
#
#trajectory_1 = (data_set_without_alignment[data_set_without_alignment.batch == 59].loc[:,variable_plot1]
#                    .reset_index(drop =True))
#trajectory_reference_1 = (data_set_without_alignment[data_set_without_alignment.batch == reference_batch]
#                        .loc[:,variable_plot1]
#                        .reset_index(drop =True))
#trajectory_2 = (data_set_without_alignment[data_set_without_alignment.batch == 60].loc[:,variable_plot1]
#                    .reset_index(drop =True))                        
#plt.plot(trajectory_1)
#plt.plot(trajectory_reference_1)
#plt.plot(trajectory_2)
#plt.show()

#%%
y = (unusual_steam_behaviour_1 + unusual_steam_behaviour_2)
y

#abnormalSteamBatches = [ 2,   6,  15,  26,  40,  41,  47,  48,  57,  62,  77,  78,  79,
#                            83,  86,  87,  89,  90,  94,  96,  97, 101, 102, 105, 107, 111,
#                            113, 118, 122, 130, 132, 133, 136, 139, 156, 158, 165, 166, 171,
#                            174, 177, 185, 186, 193, 212, 223, 228, 229, 230, 232, 235, 240, 246]
#                            
#len(abnormalSteamBatches)
#
#z = [i for i in abnormalSteamBatches if i in y]

#%%
t = batch_timing_new[batch_timing_new.batch.isin(unusual_stirrer_behaviour)].iloc[:,[1,3]]
t
#%%
t.to_csv(os.path.join(pathToOutput, 'batches_unusual_stirrer_peak.csv'))