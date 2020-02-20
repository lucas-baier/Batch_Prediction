# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 08:47:20 2017

@author: delubai
"""
import sys
sys.path.append('C:\project-data\my-codes\Spyder_workspace\MCPA_batch_data')

import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
import MCPA_functions

# Use for simplified multiindex access
idx = pd.IndexSlice

# Import R packages and set up namespace
proxy = importr("proxy",  lib_loc = "C:/Users/delubai/Documents/R/win-library/3.3")
DTW = importr("dtw", lib_loc = "C:/Users/delubai/Documents/R/win-library/3.3")

rpy2.robjects.numpy2ri.activate()
R = rpy2.robjects.r

# Set path of datasets
path_to_data_set="C://project-data/my-data-sets/data_new_processed/"
path_to_model = "C:/project-data/my-data-sets/Online_DTW_Data/"


#%%

class LoadData():
    """
    Class which loads the availabe batch data and the corresponding time stamps and can return slices of the 
    database    
    """
    def __init__(self):
        self._batch_timing = pd.read_csv(os.path.join(path_to_data_set,'batch_timing_all.csv'))
        self._data_set = pd.read_csv(os.path.join(path_to_data_set,'processed_all.csv'))
        self._data_set = self._data_set.rename(columns = {'Unnamed: 0':'time_absolut'})
         
    def get_batch_data(self, batch_no):
        """
        Function which returns the batch data of a given batch for the important
        process phases
        
        Args:
            batch_no (int): number of batch for which data is needed
            
        Returns:
            pd.DataFrame: returns a DataFrame containing the variable 
                            measurements for this batch
            pd.DataFrame: returns a DataFrame with the timestamps of the process
        """
        variables_values = self._data_set[self._data_set.batch == batch_no]
        timing_values = self._batch_timing[self._batch_timing.batch == batch_no]
        duration_batch_3_phases = timing_values.iloc[:,-3:].sum(axis = 1).values[0]
        variables_values = variables_values[variables_values['time_rel_batch, h'] < duration_batch_3_phases]
        return(variables_values, timing_values)

#%%

class LoadModel():
    """
    Class which loads the model learned with the dataset of good batches
    """
    def __init__(self):
        self.principal_components = pd.read_pickle(os.path.join(path_to_model, 'principal_components.pkl'))
        self.mean = pd.read_pickle(os.path.join(path_to_model, 'mean.pkl'))
        self.std = pd.read_pickle(os.path.join(path_to_model, 'std.pkl'))
        self.SPE_confidence_limits = pd.read_pickle(os.path.join(path_to_model, 'SPE_confidence_limits.pkl'))

#%%

class BatchData():
    """
    Class which allows to access batch data and retrieve specific phases of 
    this batch
    """    
    
    def __init__(self, batch_no, variables_data, timing_data):
        self._batch_no = batch_no
        self._variables_data = variables_data.reset_index(drop = True).sortlevel(level = 1, axis = 1)
        self._timing_data = timing_data.set_index('batch', drop = True).loc[:,'start_vacuum':]
        self._variables_data_clean = self._variables_data.drop(['time_absolut','batch', 'time_rel, h', 'time_rel_batch, h'],
                                                             axis = 1)

    def calc_phase_points(self):
        """
        Function which calculates the phase transition points for a batch    
        """  
        pre_steam = self._timing_data.loc[:,'duration_pre_steam, h'].values[0]
        steam = pre_steam + self._timing_data.loc[:,'duration_steam_dist, h'].values[0]
        post_steam = steam + self._timing_data.loc[:,'duration_post_steam, h'].values[0]
        vacuum = post_steam + self._timing_data.loc[:,'duration_vac_dist, h'].values[0]
        self._phase_points_cum = pd.Series([pre_steam, steam, post_steam, vacuum],
                                          index = ['pre_steam','steam', 'post_steam', 'vacuum'],
                                            name = 'phase points cumulated')
    
    def get_phase(self, phase):
        """
        Function which returns the batch data of a specific process phase     
        
        Args:
            phase (int): number of phase which should be returned, from 1 to 4
            
        Returns:
            pd.DataFrame: returns a DataFrame containing only the variable 
                            measurements of the process phase
        """                
        self.calc_phase_points()        
        if phase == 1:
            result = self._variables_data[(self._variables_data['time_rel_batch, h'] <
                                            self._phase_points_cum.loc['pre_steam'])]                            
        if phase == 2:
            result = self._variables_data[(self._variables_data['time_rel_batch, h'] >
                                            self._phase_points_cum.loc['pre_steam']) & 
                                         (self._variables_data['time_rel_batch, h'] <
                                            self._phase_points_cum.loc['steam'])]   
        if phase == 3:
            result = self._variables_data[(self._variables_data['time_rel_batch, h'] >
                                            self._phase_points_cum.loc['steam']) & 
                                         (self._variables_data['time_rel_batch, h'] <
                                            self._phase_points_cum.loc['post_steam'])] 
        if phase == 4:
            result = self._variables_data[(self._variables_data['time_rel_batch, h'] >
                                            self._phase_points_cum.loc['post_steam'])]
        result = result.drop(['time_absolut','batch', 'time_rel, h', 'time_rel_batch, h'],
                                                             axis = 1)                                    
        return (result)

    def get_batch_no(self):
        return(self._batch_no)
        
    def get_variables_data(self):
        return(self._variables_data)
        
    def get_timing_data(self):
        return(self._timing_data)
        
    def get_variables_data_clean(self):
        return(self._variables_data_clean)
        


#%%

class BatchDataWarpingOnline(BatchDataWarping):
    """
    Class for the execution of the online DTW algorithm. 
    
    Uses the implementation of the DTW algorithm in R
    """
    def __init__(self, batch_no, variables_data, timing_data, reference_batch):
        BatchDataWarping.__init__(self, batch_no, variables_data, timing_data,
                                  reference_batch)        
        
    def get_current_phase_no(self):
        """
        Function which returns the number of the current batch phase        

        Returns:
            int: returns a number between 1 and 4
        """          
        self.calc_phase_points()
        list_values = list(self._phase_points_cum.values)        
        result = next(i for i,v in enumerate(list_values) if v > self._variables_data.iloc[-1,:].loc['time_rel_batch, h'])      
        return(result+1)
                    
    def apply_warping_online(self):
        """ 
        Function defines necessary DTW function for R package and executes 
        the function with current phase for an online alignment of the batch       

        Returns:
            pd.DataFrame: returns the aligned query batch
        """        
        robjects.r('''
        # create a function `apply_dtw` in R
        # open.end is set to true for alignment without end-constraint
        apply_dtw <- function(query, reference) {
            stepPatternDTW = symmetricP05        
            alignment = dtw(query,
                            reference,
                            dist.method = 'Euclidean',
                            step.pattern = stepPatternDTW,
                            keep = TRUE,
                            open.end = TRUE
                            )
            return(alignment)
        }
        ''')
        apply_dtw_online = robjects.globalenv['apply_dtw']
        query = self.get_phase(self.get_current_phase_no())
        reference = self._reference_batch.get_phase(self.get_current_phase_no())    
        query_without_foaming_signal = query.drop(['LA20162-2'], axis = 1).values
        reference_without_foaming_signal = reference.drop(['LA20162-2'], axis = 1).values         
        query = self.get_phase(self.get_current_phase_no()).values       
        alignment = apply_dtw_online(query_without_foaming_signal, reference_without_foaming_signal)
        return(self.warp_alignment_online(alignment, query))        
    
    def apply_warping(self, query, reference):
        """
        Function for the warping of previous phases of the batch. No online 
        alignment needed, since phase is already known completely 
        
        Args:
            query (pd.DataFrame): DataFrame which contains the information of 
                                    the query batch 
            reference (pd.DataFrame): DataFrame which contains the information 
                                        of the reference batch

        Returns:
            alignment: returns an alignment object which can be used with DTW 
                        package       
        """        
        robjects.r('''
        # create a function `apply_dtw_old` in R
        # necessary for the warping of the previous batch phases
        apply_dtw_old <- function(query, reference) {
            stepPatternDTW = symmetricP05        
            alignment = dtw(query,
                            reference,
                            dist.method = 'Euclidean',
                            step.pattern = stepPatternDTW,
                            keep = TRUE
                            )
            return(alignment)
        }
        ''')
        apply_dtw_old = robjects.globalenv['apply_dtw_old']            
        alignment = apply_dtw_old(query, reference)
        return(alignment)  
        
    def warp_alignment_online(self, alignment, query):        
        """
        Function which returns the query warped according to the reference batch        
        
        Args:
            alignment (alignment): alignment object which contains the warping information
            query (pd.DataFrame): DataFrame which contains the query information

        Returns:
            pd.DataFrame: returns the aligned query batch
        """        
        if (query.shape[0] < 2):
            index = [1]
        else:
            index = list(R.warp(alignment))
        
        #index in python starts with 0 not with 1        
        index_python = [t-1 for t in index]
        index_int = [int(i) for i in index_python]
        query_warped = query[index_int,:]        
        pos_non_integer = [i for i, j in enumerate(index_python) if int(j) != j]
        
        #Adjust value for non int index, use mean value of the two indexes
        for i in pos_non_integer:
            index_for_mean = list([int(index_python[i]),int(index_python[i])+1])
            value_replace = np.mean(query[index_for_mean,:],
                                    axis = 0)
            query_warped[i,:] = value_replace        
        
        self._alignment_index = index_python
        query_warped = pd.DataFrame(query_warped, columns = self._variables_data_clean.columns)
        return(query_warped)
        
    def dtw_online_all_phases(self):
        """
        Function which calculates the online alignment of a batch with a 
        reference batch. Includes the execution of DTW of previous batch' 
        phases
        """                
        self._phase_1_warped = pd.DataFrame(columns = self._variables_data_clean.columns)        
        self._phase_2_warped = pd.DataFrame(columns = self._variables_data_clean.columns)
              
        if (self.get_current_phase_no() > 1):        
            query = self.get_phase(1)
            query_without_foaming_signal = query.drop(['LA20162-2'], axis = 1).values
            query = query.values            
            reference_without_foaming_signal = self._reference_batch.get_phase(1).drop(['LA20162-2'], axis = 1).values
            alignment = self.apply_warping(query_without_foaming_signal,
                                           reference_without_foaming_signal)
            self._phase_1_warped = self.warp_alignment_online(alignment, query)        
        
        if (self.get_current_phase_no() > 2):                       
            query = self.get_phase(2)
            query_without_foaming_signal = query.drop(['LA20162-2'], axis = 1).values
            query = query.values            
            reference_without_foaming_signal = self._reference_batch.get_phase(2).drop(['LA20162-2'], axis = 1).values
            alignment = self.apply_warping(query_without_foaming_signal,
                                           reference_without_foaming_signal)
            self._phase_2_warped = self.warp_alignment_online(alignment, query)  
        
        current_warping = self.apply_warping_online()
        self._warping_up_to_current_moment  = pd.concat([self._phase_1_warped,self._phase_2_warped,current_warping], ignore_index = True)       
        
        
    def plot_alignment(self, var_no):
        """
        Function which plots the current alignment of the query batch     
        
        Args:
            var_no (int): number of variable which should be displayed in the
                            plot, can take values from 0 to no_of_variables-1
        """        
        if var_no > self._variables_data_clean.shape[1]-1:
            print('Error: Chosen variable number out ouf index')
        else:
            fig, ax1 = plt.subplots()
            ax1.plot(self._reference_batch._variables_data_clean.iloc[:,var_no].values,
                     color = 'black')
            ax1.plot(self._variables_data_clean.iloc[:,var_no].values,
                     color = 'red')
            ax1.plot(self._warping_up_to_current_moment.iloc[:,var_no].values,
                     color = 'orange')
            ax1.title('Batch {}, var {}'.format(self._batch_no,
                      self._variables_data_clean.iloc[:,var_no].name))
            plt.plot()
            
    def get_warping_up_to_current_moment(self):
        return(self._warping_up_to_current_moment)

#%%

class OnlineMPCABatch():
    """
    Class for the calculation of the Squared Prediction Error (SPE) and the 
    scores with the current batch information
    """
    def __init__(self, data, batch_no, principal_components, model_mean, model_std):
        self._data = data
        self._batch_no = batch_no        
        self._principal_components = principal_components.sortlevel(axis = 1)
        self._data_without_foaming_signal = data.drop('LA20162-2', axis = 1)
        self._model_mean = model_mean.sortlevel()
        self._model_std = model_std.sortlevel()        
        
    def scale_values(self):
        """
        Function which scales the current batch data with the mean and standard
        deviation from the data that was used for model training
        """        
        data_transformed = self._data_without_foaming_signal.stack()        
        mean_adapted = self._model_mean.loc[list(data_transformed.index.levels[0])]
        std_adapted = self._model_std.loc[list(data_transformed.index.levels[0])]
        scaled = (data_transformed - mean_adapted)/(std_adapted)
        self._scaled = scaled        
        
    def calculate_online_values(self):
        """
        Function which calculates the SPE, the SPE per variable, and the score
        with the current batch data
        """                
        i = len(self._data.index)
        scores_columns = self._principal_components.index
        self._current_time_index = self._data.index[-1]        
        
        repeat = len(self._principal_components.columns.levels[0]) - i
        current_dev = self._scaled.loc[self._current_time_index]
        dev = np.tile(current_dev, repeat)
        batch_completed = np.concatenate((self._scaled, dev))
        
        self._scaled_complete = pd.Series(batch_completed,
                                         index = self._principal_components.columns)
        self._scores = pd.Series(np.dot(self._scaled_complete, self._principal_components.T),
                                index = scores_columns,
                                name = self._batch_no)
        
        self._projection = pd.Series(np.dot(self._scores, self._principal_components),
                                    index = self._principal_components.columns,
                                    name = self._batch_no)
        
        self._SPE_var = pd.Series(((self._scaled.loc[self._current_time_index] - 
                                    self._projection.loc[self._current_time_index])**2),
                                    index = self._scaled.index.levels[1],
                                    name = self._batch_no)
        
        self._SPE = pd.Series(self._SPE_var.sum(),
                             index = [self._current_time_index],
                             name = self._batch_no)
                             
    def get_SPE(self):
        return(self._SPE)

    def get_SPE_var(self):
        return(self._SPE_var)

    def get_scores(self):
        return(self._scores)

    def get_current_time_index(self):
        return(self._current_time_index)                             
        
#%%
             
class MPCABatchResult():
    """
    Class for the collection of the results of the online algorithms
    
    Needs to be extended as soon as a new measurement from the batch is available
    """
    def __init__(self, batch_no):
        self._batch_no = batch_no        
        self._SPE = pd.Series()
        self._SPE_var = pd.Series()
        self._scores = pd.Series()
        self._out_of_limits = pd.Series()
        
    def append(self, batch_online):
        """
        This function adds the current value for the Squared Prediction Error, the 
        Squared Prediction Error per Variable and the current value to the 
        MPCABatchResult object
        
        Args:
            batch_online (OnlineMPCABatch): object which contains the current 
                                                Error values and scores     
        """
        self._SPE = self._SPE.append(batch_online.get_SPE())
        
        time = batch_online.get_current_time_index()
        var = batch_online.get_SPE_var().index
        pc = batch_online.get_scores().index
        index_SPE_var = pd.MultiIndex.from_product([time, var],
                                                   names = ['time', 'variables'])
                                          
        index_scores = pd.MultiIndex.from_product([time, pc],
                                                  names = ['time', 'principal_component'])
                                       
        SPE_var = batch_online.get_SPE_var()
        SPE_var.index =  index_SPE_var
        self._SPE_var = self._SPE_var.append(SPE_var)        
        self._SPE_var.index = pd.MultiIndex.from_tuples(self._SPE_var.index)
        
        scores = batch_online.get_scores()
        scores.index = index_scores
        self._scores = self._scores.append(scores)
        self._scores.index = pd.MultiIndex.from_tuples(self._scores.index) 
    
    def check_control_values(self, confidence_limits):
        """
        This function compares the current SPE value with the confidence limits
        and prints a warning if the value leaves its limits
        
        Args:
            confidence_limits (pd.Series): contains the confidence limits for 
                                            every time step
        """        
        index = self._SPE.index[-1]
        current_SPE = self._SPE.iloc[-1]
        confidence_value = confidence_limits[confidence_limits.index == index].values[0]
        
        if current_SPE > confidence_value:
            print('SPE left confidence value. Foaming alarm! Time = {}'.format(index))
        
        boolean = current_SPE > confidence_value
        self._out_of_limits = self._out_of_limits.append(pd.Series(boolean,
                                                                   index = [index],
                                                                   name = self._batch_no))
        
    def plot_SPE_and_confidence_limits(self, confidence_limits):
        """
        Function which plots the SPE trajectory and the corresponding confidence
        limits
        """        
        batch_SPE = self._SPE[self._SPE.index.duplicated(keep = 'last') == False].sort_index()        
        last_index = batch_SPE.index[-1]
        confidence_trajectory = confidence_limits.loc[:last_index]        
        
        fig, ax1 = plt.subplots()
        ax1.plot(confidence_trajectory.index,
                 confidence_trajectory.values,
                 color = 'cornflowerblue',
                 label = '99% confidene limit')        
        ax1.plot(batch_SPE.index,
                 batch_SPE.values,
                 color = 'mediumblue',
                 label = 'SPE')
 
        plt.legend()
        plt.title('Squared Prediction Error for batch {}'.format(self._batch_no,))
        plt.plot()        
        
    def get_SPE(self):
        return(self._SPE)
        
    def get_SPE_var(self):
        return(self._SPE_var)
        
    def get_scores(self):
        return(self._scores)
        
    def get_out_of_limits(self):
        return(self._out_of_limits)
#%%
     
class BatchDataWarping(BatchData):
    """
    Class which allows to warp query batch data to a reference batch and 
    therefore create a batch with the same duration as the reference batch
    """    
    def __init__(self, batch_no, variables_data, timing_data, reference_batch):
        BatchData.__init__(self, batch_no, variables_data, timing_data)        
        self._reference_batch = reference_batch
        
    def apply_warping(self):
        """
        Function which defines a function in R which uses the DTW (Dynamic Time
        Warping) package. This package is used for the alignment of all batches.      

        Returns:
            alignment: returns an alignment object
        """           
        robjects.r('''
        # create a function `apply_dtw` in R
        apply_dtw <- function(query, reference) {
            stepPatternDTW =  symmetric2 #symmetricP05 #     
            alignment = dtw(query,
                            reference,
                            dist.method = 'Euclidean',
                            step.pattern = stepPatternDTW,
                            keep = TRUE
                            )
            return(alignment)
        }
        ''')
        apply_dtw = robjects.globalenv['apply_dtw']
        query = self._variables_data_clean.drop(['LA20162-2'], axis =1).values
        reference = self._reference_batch._variables_data_clean.drop(['LA20162-2'], axis = 1).values
        alignment = apply_dtw(query, reference)
        return(alignment)        

    def warp_alignment(self):
        """
        Function which returns warped query batch       

        Returns:
            pd.DataFrame: returns the warped query batch
        """          
        alignment = self.apply_warping()        
        index = list(R.warp(alignment))
        #index in python starts with 0 not with 1        
        index_python = [t-1 for t in index]
        index_int = [int(i) for i in index_python]
        query_warped = self._variables_data_clean.as_matrix()[index_int,:]        
        pos_non_integer = [i for i, j in enumerate(index_python) if int(j) != j]
        
        #Adjust value for non int index, use mean value of the two indexes
        for i in pos_non_integer:
            index_for_mean = list([int(index_python[i]),int(index_python[i])+1])
            value_replace = np.mean(self._variables_data_clean.as_matrix()[index_for_mean,:],
                                    axis = 0)
            query_warped[i,:] = value_replace        
        
        self._alignment_index = index_python
        query_warped = pd.DataFrame(query_warped, columns = self._variables_data_clean.columns)
        return(query_warped)

    def plot_alignment(self, var_no):
        """
        Function which plots the current alignment of the query batch     
        
        Args:
            var_no (int): number of variable which should be displayed in the
                            plot, can take values from 0 to no_of_variables-1
        """           
        if var_no > self._variables_data_clean.shape[1]-1:
            print('Error: Chosen variable number out ouf index')
        else:
            fig, ax1 = plt.subplots()
            ax1.plot(self._reference_batch._variables_data_clean.iloc[:,var_no].values,
                     color = 'grey',
                     label = 'Reference batch',
                     lw = 1)
            ax1.plot(self._variables_data_clean.iloc[:,var_no].values,
                     color = 'lightsteelblue',
                     label = 'Query batch',
                     lw = 1)
            ax1.plot(self.warp_alignment().iloc[:,var_no].values,
                     color = 'steelblue',
                     label = 'Query batch aligned')
            plt.title('Batch {}'.format(self._batch_no))                     
            ax1.set_ylim(0,1.8)
            ax1.set_xlabel('Time [min]')
            plt.legend()
            plt.plot()
        
#%%
index_training_batches = MCPA_functions.index_training_batches
index_test_batches = MCPA_functions.index_test_batches
index_foaming_batches = MCPA_functions.index_foaming_batches

unusual_stirrer_behaviour = [37, 262, 133, 259, 283, 284, 127, 166, 193, 268, 163, 167, 246, 261,
                             239, 255, 256, 245, 267, 74, 149, 182, 183, 185, 174, 181, 200, 124,
                             165, 90, 98, 264, 260, 265, 164, 213, 214, 296, 312, 201, 231, 237, 
                             202, 203, 236, 282, 170, 194, 184, 125, 168, 126, 169]


# Set the batch numbers
ref_batch_no = 108        
  
# Load the necessary batch data
data = LoadData()
ref_data, ref_timing = data.get_batch_data(ref_batch_no)
reference_batch =  BatchData(ref_batch_no, ref_data, ref_timing)
#174
for batch_no in [245]:
    data_batch, timing_batch = data.get_batch_data(batch_no)
    
    Test = BatchDataWarping(batch_no, data_batch, timing_batch, reference_batch)
    
    Test.apply_warping()
    Test.warp_alignment()
    Test.plot_alignment(0)