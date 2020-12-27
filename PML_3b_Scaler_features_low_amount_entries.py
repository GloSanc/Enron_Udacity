#!/usr/bin/env python
# coding: utf-8

# # 3.b. Scaler behavior on features with low amount of entries

# # 3.b.1. Dataframe creation with features with low amount of entries and big amount of entries

# In[9]:


#coding: utf-8 

### Import some packages and modules to be used later
import pickle
import pandas as pd
import numpy as np

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from PML_1_data_adq import data_dict_cor 

from feature_format import featureFormat, targetFeatureSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

### Loading the dictionary containing the dataset

data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

### running the function to remove the non person entries and correct the wrong values
### 'THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 'BELFER ROBERT', 'BHATNAGAR SANJAY'

data_dict_cor (data_dict)

my_dataset = data_dict


# Defining the list of features to be loaded

features_list=[ 'loan_advances', 'director_fees',  'restricted_stock_deferred', 
                'deferral_payments', 'deferred_income','long_term_incentive',
               'from_messages', 
               'total_payments','total_stock_value',
                'poi' ]
        
# Creating the data frame from the dictionary with the features list defined and replacing Nan by 0

data_df = pd.DataFrame.from_dict(data_dict, orient='index')

data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df= data_df.loc[ :,(features_list )]  

data_df= data_df.replace('NaN', 0.0)
round(data_df.describe(),0)

data_df=data_df
data_df


# In[2]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(data_df , vars=['loan_advances', 'director_fees',  'restricted_stock_deferred', 
                'deferral_payments', 'deferred_income','long_term_incentive',
               'from_messages', 
               'total_payments','total_stock_value'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# # 3.b.2. MinMaxScaler Scaler result

# In[3]:


def scale_mms(input_df):
    """
    Scale/Normalize all the feature columns in the given data frame except 'poi'
    Returns the scaled df
    """
    # clean
    temp_df = input_df[['poi']] # for repatch
    input_df = input_df.loc[:, input_df.columns.difference(['poi'])] # remove them if existing

    # scale
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler().fit(input_df)
    MinMaxScaler()
    input_df.loc[:]=transformer.transform(input_df)
   


    # repatch
    input_df = pd.concat([input_df, temp_df],axis=1, sort=False)

    return input_df

data_df_mms = scale_mms(data_df)

data_df_mms=round(data_df_mms,2)

round(data_df_mms.describe(),2)


# In[4]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(data_df_mms , vars=['loan_advances', 'director_fees',  'restricted_stock_deferred', 
                'deferral_payments', 'deferred_income','long_term_incentive',
               'from_messages', 
               'total_payments','total_stock_value'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# # 3.b.3.  Robust Scaler result

# In[5]:


def scale_rob(input_df):
    """
    Scale/Normalize all the feature columns in the given data frame except 'poi'
    Returns the scaled df
    """
    # clean
    temp_df = input_df[['poi']] # for repatch
    input_df = input_df.loc[:, input_df.columns.difference(['poi'])] # remove them if existing

    # scale
    from sklearn.preprocessing import RobustScaler
    transformer = RobustScaler().fit(input_df)
    RobustScaler()
    input_df.loc[:]=transformer.transform(input_df)
   
   


    # repatch
    input_df = pd.concat([input_df, temp_df],axis=1, sort=False)

    return input_df

data_df_rob = scale_rob(data_df)

data_df_rob=round(data_df_rob,2)

round(data_df_rob.describe(),2)


# In[6]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(data_df_rob, vars=['loan_advances', 'director_fees',  'restricted_stock_deferred', 
                'deferral_payments', 'deferred_income','long_term_incentive',
               'from_messages', 
               'total_payments','total_stock_value'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# # 3.b.3. Normalize Scaler result

# In[7]:


def scale_nor(input_df):
    """
    Scale/Normalize all the feature columns in the given data frame except 'poi'
    Returns the scaled df
    """
    # clean
    temp_df = input_df[['poi']] # for repatch
    input_df = input_df.loc[:, input_df.columns.difference(['poi'])] # remove them if existing

    # scale
    from sklearn.preprocessing import Normalizer
    transformer = Normalizer().fit(input_df)
    Normalizer()
    input_df.loc[:]=transformer.transform(input_df)
   


    # repatch
    input_df = pd.concat([input_df, temp_df],axis=1, sort=False)

    return input_df

data_df_nor = scale_nor(data_df)

data_df_nor=round(data_df_nor,2)

round(data_df_nor.describe(),2)


# In[8]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(data_df_nor, vars=['loan_advances', 'director_fees',  'restricted_stock_deferred', 
                'deferral_payments', 'deferred_income','long_term_incentive',
               'from_messages', 
               'total_payments','total_stock_value'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])

