import numpy as np
import pandas as pd
from scipy import stats


def numerical_analysis(data,perc=[.05,.1,.25,.3,.4,.5,.6,.75,.8,.9,.95],trim=.05,group=None,fill_na='na'):
    
    '''data: dataframe
       perc: desired percentiles to be calculated
       trim: size of tails to be trimmed when calculating the trimmed mean
       group: grouping variables (must be passed as a list)
       fill_na: string to replace na's
    '''
    
    import time
    start_time = time.time() ## getting initial time of execution
    
    ## verifying if the user wants a groupy or not
    if group == None:
        data_aux = data.select_dtypes(include=['number']).describe(percentiles=perc).T  ## describe of numeric columns
        count_data = data.select_dtypes(include=['number']).apply(lambda x:pd.Series({'na':(x.isna().sum()),
                                                                                      'negatives':(x<0).sum(),
                                                                                      'zeros':(x==0).sum(),
                                                                                      'positives':(x>0).sum(),
                                                                                      'na_perc':(x.isna().sum())/len(x),
                                                                                      'neg_perc':((x<0).sum())/len(x),
                                                                                      'zero_perc':((x==0).sum())/len(x),
                                                                                      'pos_perc':((x>0).sum())/len(x),
                                                                                      'trim_mean':stats.trim_mean(x.dropna(),trim)})).T ## count of numeric columns
        numeric_df = pd.merge(data_aux,count_data,left_index=True,right_index=True) ## joining both dataframes    
    
    
    else:
        columns = data.select_dtypes(include=[np.number]).columns.tolist() 
        n_variables = range(len(group),len(group)+len(columns)) ## range to iterate through variables of interest
        
        ## for loop to get all the variables needed (including the grouping variables)
        for var in group:
            columns.append(var)
            
            
        data[group] = data[group].fillna(fill_na) ## replacing NA's on grouping variables (so it doesn't get ignored when grouping is made)
               
        data_aux = data[columns].groupby(group).describe(percentiles=perc).T.unstack(level=0).T.rename_axis(group+['variable'])  ## describe of numeric columns
        count_data = data[columns].groupby(group).agg(lambda x:{'na':(x.isna().sum()),
                                                                'negatives':(x<0).sum(),
                                                                'zeros':(x==0).sum(),
                                                                'positives':(x>0).sum(),
                                                                'na_perc':(x.isna().sum())/len(x),
                                                                'neg_perc':((x<0).sum())/len(x),
                                                                'zero_perc':((x==0).sum())/len(x),
                                                                'pos_perc':((x>0).sum())/len(x),
                                                                'trim_mean':stats.trim_mean(x.dropna(),trim)}).reset_index() ## count of numeric columns
        
        ## auxiliary variables
        n_combinations = range(0,count_data.shape[0])
        n_groupvar = len(group)
        data_list = []
        
        for i in n_combinations: ## loop to go through every combination of grouping variables
            for j in n_variables: ## loop to extract the values of count_data
                a = []
                for q in range(0,n_groupvar): ## loop needed to get all the group values
                    a.append(count_data.iloc[i,q])
                a.append(count_data.T.index[j])
                a.extend(list(count_data.iloc[i,j].values()))
                data_list.append(a)
    
        colnames = group + ['variable','na','negatives','zeros','positives','na_perc','neg_perc','zero_perc','pos_perc','trim_mean']
        count_data = pd.DataFrame(data=data_list,columns=colnames).set_index(group+['variable'])
        numeric_df = pd.merge(data_aux,count_data,left_index=True,right_index=True) 
     
   
    cols = numeric_df.columns.tolist() ## auxiliary list to return the dataframe in the desired order
    print("Runtime: {} seconds.".format(round(time.time() - start_time,4)))  
    return (numeric_df[cols[:2]+cols[-1:]+cols[2:-1]])


def categorical_analysis(data,group=None,unique_cut=.9,fill_na='na'):
    
    '''data: dataframe
       group: grouping variables (must be passed as a list)
       unique_cut: high cardinality check (if a variable has more than a given number of unique categories it's frequencies won't be calculated)
       fill_na: string to replace na's
    '''
    
    import time
    start_time = time.time() ## getting initial time of execution
    
    
    ## first step is to exclude every categorical variable with a high percentage of unique values
    unique_df = (data.select_dtypes(exclude=[np.number]).nunique()/data.select_dtypes(exclude=[np.number]).count()).to_frame('unique_pct')
    columns = unique_df[unique_df.unique_pct <= unique_cut].index.tolist()
    
    
    if group == None:
    
        count_cat = data[columns].apply(lambda x:x.value_counts(dropna=False)).T.stack().to_frame('n').rename_axis(['variable','category'])
        count_cat['n_pct'] = count_cat['n'].div(count_cat.groupby('variable')['n'].transform('sum')) ## creating column of percentages
    
    else:
        data[group] = data[group].fillna(fill_na,downcast=False)
        aux_cols = [var for var in columns if var not in group]
        
        count_cat = pd.DataFrame()
        
        for col in aux_cols:
            aux_df = data[columns].groupby(group)[col].value_counts(dropna=False).to_frame('n').assign(variable=col).reset_index().set_index(group+['variable']).rename(columns={col:'category'})
            aux_df['n_pct'] = aux_df['n'].div(aux_df.groupby(group)['n'].transform('sum'))
            count_cat = count_cat.append(aux_df)
            
            
    print("Runtime: {} seconds.".format(round(time.time() - start_time,4)))      
    return(count_cat)
