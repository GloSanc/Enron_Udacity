#!/usr/bin/env python
# coding: utf-8

# # 3.a.  Features: Create and Scale features
# 
# We create features and we scale them
# Features creation: The step are define the function to create features by mathematical operations, define the function to create features as a ration between existing features, define the code to create some features with PCA. 
# Define the features list for each features creation.
# Then create features by mathematical operations, create the features with PCA, pass the scaler, create the features ratio, pass scaler again and check which are the features chosen by Kbest 
# 
# 

# In[2]:


#coding: utf-8 

### Import some packages and modules to be used later
import pickle
import pandas as pd
import numpy as np

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

#from PML_1_data_adq import data_dict_cor 

from feature_format import featureFormat, targetFeatureSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC


# # 3.a.1. Create new features 
# Function definition to create some features by mathematical operations and integrate them in the dataset

# In[3]:


#Function to define new features
def create_feat (features_original_list,features_amendm_list):
    
    
    import pickle
    from feature_format import featureFormat, targetFeatureSplit
    #from tester import dump_classifier_and_data
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
   
    
# Loading the dictionary containing the dataset

    data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
   
    
# Right data to be corrected in the dataset

    data_dict['BELFER ROBERT']={'salary': 'NaN',
     'to_messages': 'NaN',
     'deferral_payments': 'NaN',
     'total_payments': 3285,
     'loan_advances': 'NaN',
     'bonus': 'NaN',
     'email_address': 'NaN',
     'restricted_stock_deferred': -44093,
     'deferred_income':  -102500,
     'total_stock_value': 'NaN',
     'expenses': 3285,
     'from_poi_to_this_person': 'NaN',
     'exercised_stock_options': 'NaN',
     'from_messages': 'NaN',
     'other': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'poi': False,
     'long_term_incentive': 'NaN',
     'shared_receipt_with_poi': 'NaN',
     'restricted_stock': 44093,
     'director_fees': 102500}

    data_dict['BHATNAGAR SANJAY']= {'salary': 'NaN',
     'to_messages': 523,
     'total_stock_value': 15456290,
     'deferral_payments': 'NaN',
     'total_payments': 137864,
     'loan_advances': 'NaN',
     'bonus': 'NaN',
     'email_address': 'sanjay.bhatnagar@enron.com',
     'restricted_stock_deferred': -2604490,
     'deferred_income': 'NaN',
     'expenses': 137864,
     'from_poi_to_this_person': 0,
     'exercised_stock_options': 15456290,
     'from_messages': 29,
     'other': 'NaN',
     'from_this_person_to_poi': 1,
     'poi': False,
     'long_term_incentive': 'NaN',
     'shared_receipt_with_poi': 463,
     'restricted_stock': 2604490,
     'director_fees': 'NaN'}
    
    
    ### Deleting the entries in the dictionary
    ### Entries to be deleted in the dataset
    list_out=['TOTAL','THE TRAVEL AGENCY IN THE PARK']

    for name in list_out:
        data_dict.pop(name,0)
        
        
# New features

    for v in data_dict.values():
        salary=v["salary"]
        bonus = v["bonus"]
        expenses = v["expenses"]
        other = v["other"]
        
        long_term_incentive = v["long_term_incentive"]
        exercised_stock_options =v["exercised_stock_options"] 
        
        total_payments = v["total_payments"]
        total_stock_value = v["total_stock_value"]

        shared_receipt_with_poi = v["shared_receipt_with_poi"]
        from_poi_to_this_person = v["from_poi_to_this_person"]
        from_this_person_to_poi = v["from_this_person_to_poi"]
 
                     
        v["incentives"] =  (float(  bonus) + float( long_term_incentive) + float(  exercised_stock_options) if
                            bonus not in [0, "NaN"] and long_term_incentive not in [0, "NaN"] 
                            and  exercised_stock_options not in [0, "NaN"]  else 0.0)
        
        incentives= v["incentives"] 
            
        v["total_money"] =  (float( total_payments) + float( total_stock_value) if
                            total_payments not in [0, "NaN"] and total_stock_value
                           not in [0, "NaN"] else 0.0)
        
        total_money=  v["total_money"]  
        
     
              
             
    features_original_list.append("incentives") 
   
    features_original_list.append("total_money")
    
    
    my_dataset = data_dict

# data frame
    data_df = pd.DataFrame.from_dict(data_dict, orient='index')
    data_df= data_df.loc[ :,(features_amendm_list )]  

    data_df= data_df.replace('NaN', 0.0)
    data_df=round(data_df,2)

    #print('data frame shape',data_df.shape)
    return data_df



# Function definition to create some features as ratio between existing featrues and integrate them in the dataset

# In[4]:


#Function to define new features
def create_ratios (features_original_list,features_amendm_list,data_dict):
    
    
    import pickle
    from feature_format import featureFormat, targetFeatureSplit
    #from tester import dump_classifier_and_data
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import warnings
    warnings.filterwarnings('ignore')
          
        
# New features

    for v in data_dict.values():
        salary=v["salary"]
        bonus = v["bonus"]
        expenses = v["expenses"]
        other = v["other"]
        
        long_term_incentive = v["long_term_incentive"]
        exercised_stock_options =v["exercised_stock_options"] 
        
        total_payments = v["total_payments"]
        total_stock_value = v["total_stock_value"]

        shared_receipt_with_poi = v["shared_receipt_with_poi"]
        from_poi_to_this_person = v["from_poi_to_this_person"]
        from_this_person_to_poi = v["from_this_person_to_poi"]
 

        v["incentives"] =  (float(  bonus) + float( long_term_incentive) + float(  exercised_stock_options) if
                            bonus not in [0, "NaN"] and long_term_incentive not in [0, "NaN"] 
                            and  exercised_stock_options not in [0, "NaN"]  else 0.0)
        
        incentives= v["incentives"] 


            
        
        v["total_money"] =  (float( total_payments) + float( total_stock_value) if
                            total_payments not in [0, "NaN"] and total_stock_value
                           not in [0, "NaN"] else 0.0)
        
        total_money=  v["total_money"]  
        
        v["salary_ratio"] =  (float( salary) / float( total_money) if
                            salary not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
        
        salary_ratio=  v["salary_ratio"]  
        
        v["bonus_ratio"] =  (float( bonus) / float( total_money) if
                           bonus not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
        
        bonus_ratio=  v["bonus_ratio"]         
        
        v["expenses_ratio"] =  (float( expenses) / float( total_money) if
                           expenses not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
        
        expenses_ratio=  v["expenses_ratio"]            
        
        v["other_ratio"] =  (float( other) / float( total_money) if
                          other not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
        
        other_ratio=  v["other_ratio"]    
        
        v["long_term_incentive_ratio"] =  (float( long_term_incentive) / float( total_money) if
                               long_term_incentive not in [0, "NaN"] and total_money
                                not in [0, "NaN"] else 0.0)
        
        long_term_incentive_ratio=  v["long_term_incentive_ratio"]  
                
        v["exercised_stock_options_ratio"] =  (float( exercised_stock_options) / float( total_money) if
                          exercised_stock_options not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
        
        exercised_stock_options_ratio=  v["exercised_stock_options_ratio"]  
        
        v["incentives_ratio"] =  (float(  incentives) / float( total_money) if
                          incentives not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
                
        incentives_ratio= v["incentives_ratio"] 
        
 
        v["total_payments_ratio"] =  (float( total_payments) / float( total_money) if
                          total_payments not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
                
        total_payments_ratio= v["total_payments_ratio"] 
        
        v["total_stock_value_ratio"] =  (float(total_stock_value) / float( total_money) if
                          total_stock_value not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
                
        total_stock_value_ratio= v["total_stock_value_ratio"]       
        
        v["from_poi_to_this_person_ratio"] =  (float(  from_poi_to_this_person) / float( shared_receipt_with_poi) if
                          from_poi_to_this_person not in [0, "NaN"] and shared_receipt_with_poi
                           not in [0, "NaN"] else 0.0)
                
        from_poi_to_this_person_ratio= v["from_poi_to_this_person_ratio"] 
        
        v["from_this_person_to_poi_ratio"] =  (float( from_this_person_to_poi) / float( shared_receipt_with_poi) if
                         from_this_person_to_poi not in [0, "NaN"] and shared_receipt_with_poi
                           not in [0, "NaN"] else 0.0)
                
        from_this_person_to_poi_ratio= v["from_this_person_to_poi_ratio"] 
              
             
    
    features_original_list.append("incentives_ratio")
 
    
    features_original_list.append("salary_ratio")
    features_original_list.append("bonus_ratio")
    
    features_original_list.append("expenses_ratio")
    features_original_list.append("other_ratio")
    
    features_original_list.append("exercised_stock_options_ratio") 

    features_original_list.append("total_payments_ratio")
    features_original_list.append("total_stock_value_ratio")     
                          
    features_original_list.append("from_poi_to_this_person_ratio")                          
    features_original_list.append("from_this_person_to_poi_ratio")     
    
    

    my_dataset = data_dict

# data frame
    data_df = pd.DataFrame.from_dict(data_dict, orient='index')
    data_df= data_df.loc[ :,(features_amendm_list )]  

    data_df= data_df.replace('NaN', 0.0)
    data_df=round(data_df,2)

    #print('data frame shape',data_df.shape)
    return data_df


# We need to specify the origina feature list and the final one to run the function 

# In[5]:


# 1. original features
features_list0=[ 'poi',
                               'salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi']

# 2. features including new after create_feat
features_cf=[ "incentives","total_money"  ]

features_list_cf=list(set(features_list0+features_cf))


# 3. features including new after create_feat and create_ratio and PCA
features_pca=[ 'incentive_f','payment_f','payment_2','payment_tt','emails_exc']

features_list_cf_pca=list(set(features_list_cf+features_pca))


# 4. features including new after create_feat and create_ratio
features_rt= ['incentives_ratio',"salary_ratio", "bonus_ratio", 
            "expenses_ratio","other_ratio",  
            "exercised_stock_options_ratio", 
            "total_payments_ratio", "total_stock_value_ratio", 
            "from_poi_to_this_person_ratio", "from_this_person_to_poi_ratio"]

features_list_cf_pca_rt=list(set(features_list_cf_pca+features_rt))


# 5. All features including new ones and excluding the ones with low amount of values
features_low=[ 'deferred_income','deferral_payments', 'loan_advances',
                    'director_fees', 'restricted_stock_deferred' ]

features_list_cl=list(set(features_list_cf_pca_rt).difference(set(features_low)))


# 6. features list without poi . The one to pass to the algo
poi=['poi']
algo_list= list(set(features_list_cl).difference(set(poi)))

# 7. features list with poi
features_list=algo_list.copy()
features_list.insert(0,'poi')


# 8. output list with poi and poi_pred
output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# 9. orginal features list without poi . The one to pass to the algo
algo_list0= list(set(features_list0).difference(set(poi)))
algo_list0= list(set(algo_list0).difference(set(features_low)))


# We run the functions to create the features and generate the dataframe and the dict 

# In[6]:


# Creating features

############
#Run create_feat (features_original_list,features_amendm_list):

data_df_cf=create_feat (features_list0,features_list_cf)
data_df_cf=round(data_df_cf,2)
data_df_cf= data_df_cf.loc[ :,(features_list_cf)]  

data_dict_cf=data_df_cf.to_dict('index')

################

# Features creation with PCA
data_df=data_df_cf

payment_f=['salary', 'bonus', 'long_term_incentive', 'other', 'expenses']
payment_2=['salary', 'other', 'expenses']
payment_tt=['salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees']
incentive_f=['bonus', 'long_term_incentive','exercised_stock_options']
stock_features=['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
emails_exc=['from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
                'shared_receipt_with_poi']

pca = PCA(n_components=1)
pca.fit(data_df[payment_2])
pcaComponents = pca.fit_transform(data_df[payment_2])

data_df['payment_2']=pcaComponents

pca = PCA(n_components=1)
pca.fit(data_df[payment_f])
pcaComponents = pca.fit_transform(data_df[payment_f])

data_df['payment_f']=pcaComponents

pca = PCA(n_components=1)
pca.fit(data_df[payment_tt])
pcaComponents = pca.fit_transform(data_df[payment_tt])

data_df['payment_tt']=pcaComponents


pca = PCA(n_components=1)
pca.fit(data_df[incentive_f])
pcaComponents = pca.fit_transform(data_df[incentive_f])

data_df['incentive_f']=pcaComponents

pca = PCA(n_components=1)
pca.fit(data_df[emails_exc])
pcaComponents = pca.fit_transform(data_df[emails_exc])

data_df['emails_exc']=pcaComponents


#data_df  = data_df .loc[ :,(features_list_cf_pca)]

data_df_pca=round(data_df,2)
data_df_pca= data_df_pca.loc[ :,(features_list_cf_pca)]  

data_dict_pca=data_df_pca.to_dict('index')


############
#Run create_ratios (features_original_list,features_amendm_list,data_dict):
data_df_cr=create_ratios (features_list_cf_pca,features_list_low,data_dict_pca)
data_df_cr=round(data_df_cr,2)
data_df_cr= data_df_cr.loc[ :,(features_list_low)]  

###########
#create final dataframe and dict without :  
#'director_fees', 'restricted_stock_deferred' , 'loan_advances' and ‘deferral_payments' 
data_df= data_df_cr.loc[ :,(features_list)] 
data_df=round(data_df,0)
data_dict=data_df.to_dict('index')
my_dataset=data_dict

data_des_t=(round(data_df.describe(),0)).transpose()
data_des_t= round((data_des_t.sort_values('std',ascending=False)),0)
data_des_t


# # 3.a.2. Features selection before scaling

# In[ ]:


def do_split(data):
    X = data.copy()
    #Removing the poi labels and put them in a separate array, transforming it
    #from True / False to 0 / 1
    y = X.pop("poi").astype(int)
    
    return X, y, 

from sklearn.linear_model import LogisticRegression

pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify', AdaBoostClassifier(random_state=42)
                 )])


#CHANGE HERE THE number of features
N_FEATURES_OPTIONS = list(range(2,32))

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

#CHANGE HERE THE DATAFRAME NAME
X, y = do_split(data_df)
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores = mean_scores.reshape(-1, len(N_FEATURES_OPTIONS))
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)
plt.figure(figsize=(12,9))

for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label)

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0, 0.4))
plt.legend(loc='upper left')

plt.show()
grid.best_estimator_


# # 3.a.3. Scaling with RobustScaler

# In[ ]:


def scale_rob(input_df):
    """
    Scale/Normalize all the feature columns in the given data frame except 'email_address' and 'poi'
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

data_rob_des_t=(round(data_df_rob.describe(),0)).transpose()
data_rob_des_t= round((data_rob_des_t.sort_values('std',ascending=False)),0)
data_rob_des_t


# In[ ]:


data_dict_rob=data_df_rob.to_dict('index')
my_dataset_rob=data_dict_rob
#my_dataset


# # 3.a.4. Features selection after Robust scaler

# In[ ]:




pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify', AdaBoostClassifier(random_state=42))])

N_FEATURES_OPTIONS = list(range(2,32))

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

#CHANGE HERE THE DATAFRAME NAME
X, y = do_split(data_df_rob)
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores = mean_scores.reshape(-1, len(N_FEATURES_OPTIONS))
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)
plt.figure(figsize=(12,9))

for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label)

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0, 0.4))
plt.legend(loc='upper left')

plt.show()
grid.best_estimator_


# In[ ]:


#Kbest after Robust scaling
data = featureFormat(my_dataset_rob, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.3, random_state=42)

from sklearn.feature_selection import SelectKBest, f_classif

kbest = SelectKBest(f_classif, k=19)
features_selected = kbest.fit_transform(features_train, labels_train)
print(features_selected.shape)

final_features = [features_list[i+1] for i in kbest.get_support(indices=True)]#i+1 as 1 in features list is 'poi'
final_features = sorted(final_features, key=lambda x: x[1], reverse=False)
print ('Features selected by SelectKBest:')
print (final_features)                                    
                                    


# # 3.a.5. Scaling with MinMax scaler

# In[ ]:


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
    transformer = MaxAbsScaler().fit(input_df)
    MinMaxScaler()
    input_df.loc[:]=transformer.transform(input_df)
   


    # repatch
    input_df = pd.concat([input_df, temp_df],axis=1, sort=False)

    return input_df

data_df_mms = scale_mms(data_df)

data_df_mms=round(data_df_mms,2)

round(data_df_mms.describe(),2)

data_dict_mms=data_df_mms.to_dict('index')
my_dataset=data_dict_mms
#my_dataset

data_mms_des_t=(round(data_df_mms.describe(),0)).transpose()
data_mms_des_t


# # 3.a.6. Features selection after MinMax scaler

# In[ ]:



pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify', AdaBoostClassifier(random_state=42))])

N_FEATURES_OPTIONS = list(range(2,32))

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

#CHANGE HERE THE DATAFRAME NAME
X, y = do_split(data_df_mms)
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores = mean_scores.reshape(-1, len(N_FEATURES_OPTIONS))
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)
plt.figure(figsize=(12,9))

for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label)

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0, 0.4))
plt.legend(loc='upper left')

plt.show()
grid.best_estimator_                    
                                    


# In[ ]:


#Kbest maxmin scaling
my_dataset=data_dict_mms
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.3, random_state=42)


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold    

kbest = SelectKBest(f_classif, k=19)
features_selected = kbest.fit_transform(features_train, labels_train)
print(features_selected.shape)

final_features = [features_list[i+1] for i in kbest.get_support(indices=True)]#i+1 as 1 in features list is 'poi'
final_features = sorted(final_features, key=lambda x: x[1], reverse=False)
print ('Features selected by SelectKBest:')
print (final_features)                                    
                                    


# # 3.a.7. Scaling with Normalizer Scaler

# In[ ]:


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

data_dict_nor=data_df_nor.to_dict('index')
my_dataset=data_dict_nor
#my_dataset

data_nor_des_t=(round(data_df_nor.describe(),3)).transpose()
data_nor_des_t


# # 3.a.8. Features selection after Normalizer Scaler

# In[ ]:



pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify', AdaBoostClassifier(random_state=42))])

N_FEATURES_OPTIONS = list(range(2,32))

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

#CHANGE HERE THE DATAFRAME NAME
X, y = do_split(data_df_nor)
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores = mean_scores.reshape(-1, len(N_FEATURES_OPTIONS))
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)
plt.figure(figsize=(12,9))

for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label)

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0, 0.4))
plt.legend(loc='upper left')

plt.show()
grid.best_estimator_

#Kbest maxmin scaling
my_dataset=data_dict_nor
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.3, random_state=42)


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold    

kbest = SelectKBest(f_classif, k=8)
features_selected = kbest.fit_transform(features_train, labels_train)
print(features_selected.shape)

final_features = [features_list[i+1] for i in kbest.get_support(indices=True)]#i+1 as 1 in features list is 'poi'
final_features = sorted(final_features, key=lambda x: x[1], reverse=False)
print ('Features selected by SelectKBest:')
print (final_features)                                    
                                    


# After this, the decision is to use Robus scaler as I decide to maintain the outlier in the data set and Robust Scaler is more adecuate to manage scalation with outlier in the dataset
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html?highlight=scaler%20outlier
# RobustScaler
# Unlike the previous scalers, the centering and scaling statistics of this scaler are based on percentiles and are therefore not influenced by a few number of very large marginal outliers. Consequently, the resulting range of the transformed feature values is larger than for the previous scalers and, more importantly, are approximately similar: for both features most of the transformed values lie in a [-2, 3] range as seen in the zoomed-in figure. Note that the outliers themselves are still present in the transformed data. If a separate outlier clipping is desirable, a non-linear transformation is required (see below).
