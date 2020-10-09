#!/usr/bin/env python
# coding: utf-8

# # Machine learning project

# 1. Data adquisition: import data, first view of data and curious findings
# 2. Data check: cleaning NaN, visualize data and check outliers
# 2. define features 
# 3. PCA chose features
# 4. try clasifiers and check scores
# 5. chose one clasifier and explain result

# The purpose of this project is to choose and algorithm and fine tune it to get identification of Person of interest in Enron fraud. It is a supervised learning as we have the solution and we could check the error of the algorithm. Once we get a set of data, the levers that we have to get a good result are the features selection and definition, the classifier and the fine tune of the classifier. 
# The key is the data quality and the data treatment before starting to submit the data set to the algorithm. We need to spend some time on data to understand what the matter is and to ensure the information we will pass to the algorithm will not induce bays, error or misleading results
# The project result is the identification of who could be considered as person of interest in the investigation of Enron fraud. At the time of the fraud investigation some financial figures from employees were reveal as well as the emails in their accounts. The idea is to use the information related to financial compensations and emails exchange to define who is and who is not a person of interest for the investigation.
# The data set contains 146 entries (Poi) and 21 features. The features in the data fall into three major types, namely financial features, email features and POI labels. 
# The features are:
# 
# 1. POI label: [‘poi’] (boolean, represented as integer)
# 
# 2. financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)
# 
# 3. email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

# # 1. DATA ADQUISITION
# 
# First thing to do is to create a dictionary with the features we need, convert that dictionary in a data frame and we check the keys and the values of one key.

# In[32]:


import os 

os.chdir(r'C:\Users\sanchez_sanc\Desktop\data_analyst\Curso\4_Machine_learning\Curso_PY_3\tools') 

import sys
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#import warnings
#warnings.filterwarnings('ignore')

### Task 1: Select what features we'll use (The first feature must be "poi")
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees']

### Loading the dictionary containing the dataset
os.chdir(r'C:\Users\sanchez_sanc\Desktop\data_analyst\Curso\4_Machine_learning\Curso_PY_3\final_project') # the directory where you want to go

sys.path.append("../Curso_PY_3/final_project/")
data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

import pandas as pd
data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df  = data_df .loc[ :,('poi','salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments', 'restricted_stock_deferred','exercised_stock_options', 
                 'restricted_stock', 'total_stock_value', 'email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]
data_df.head()


# It seams we have several 'NaN' *no information avalaible*. We need to clean those 'NaN' before start to analyse our data
# We can see that some people has not email address, but have salary and other financial figures. Lets keep that in mind for later investigation (eg. how many people without adrress mail do we have? have they any specific commmon point? is that an issue when compiling data?)
# Let's see a bit the structure, type and content of our data

# In[33]:


data_df.info()


# We have 146 people in our data with 21 posibles informations but we do not know what is the 'NaN' volume per person and per information. We need to be aware of those 'NaN' to be able to correctly interpretate the posible output of our investigation

# In[26]:


nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
round((nan_summary.sort_values('nan-proportion',ascending=False)),2)


# Lets think a bit about those 'NaN' before any decission. 
# We have a first feature Loan advances that is collect only for 3% of the people. Who they are?
# If this feature seams not relevant, we could delete from our investigation. Lets see the details of the payments

# In[47]:


loan_people = data_df.loc[:, 'loan_advances'] != 'NaN'
loan_people_df = data_df.loc[loan_people ] 
loan_people = loan_people_df.loc[:, ('poi', 'salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments')]
loan_people.head()


# We have found a couple of interesting information
# There are 4 rows
# 1.	There is one entry called TOTAL that is the sum of all values. We should remove this entry as it is not a person 
# 2.	Kenneth Lay: he got a loan advances of 81m, is nearly 100% of all the loan advances. This advance makes his salary very low. This is misleading information. We should keep that in mind later to create a feature that combine several payments as each payment by itself could be misleading. That will contribute to reduce the amount features the algorithm should analyse and avoid misleading features. We will back later on this
# 3.	There are two other people who got loan advances that are non Poi that means loan advances is not discriminant of Poi. That could indicate that is not relevant feature by itself.
# 4.	We do not have 146 useful entries but 145 as one of them is 'Total'
# 
# Now less investigate what is behind director fees, as there are only 17 entries. Who are those directors, what is their salay and stock value and the exchange of mails with poi?

# In[53]:


d_fees = data_df.loc[:, 'director_fees'] != 'NaN'
d_fees_df = data_df.loc[d_fees ] 
d_fees = d_fees_df.loc[:, ('poi', 'director_fees','salary', 'total_payments','total_stock_value',
                           'email_address', 'from_poi_to_this_person', 'from_this_person_to_poi')]
d_fees


# This is a interesting feature as all the people that got director fees is considered as no Poi, then we will keep this feature as could help the algorithm to discriminate
# 
# We can see that despite having director fees, there are two person for which total payment is NaN. 
# Lets see what is behind the people with total payment NaN
# 1. William Power: NaN in this case should be 0 as the director fees are neutralized by deferred income. 
# Note: Powers — a lawyer and professor — was tapped by Enron’s board to lead a special investigation to determine what went wrong and why.
# 
# 2. Ronnie Chan: NaN in this case should be 0 as the director fees are neutralized by deferred income. 
# Note: Ronnie Chan Chi-chung is a Hong Kong billionaire businessman. He was a director of Enron Corporation and a member of its audit committee when it filed for bankruptcy as a result of fraud.
# 
# Lets see a bit what is behind the entries where total payment is NaN. 

# In[29]:


t_payms = data_df.loc[:, 'total_payments'] == 'NaN'
df_t_payms = data_df.loc[t_payms]
df_t_payms


# This is a interesting feature as all the people that total payment NaN is considered as no Poi, then we will keep this feature as could help the algorithm to discriminate

# Regarding the mails, we see that there are 111 entries with email address but only 86 entries with values on the messages categories. Let’s see who the people without email is, confirm that they have NaN in message and if they got payments and stock

# In[52]:


n_email_people = data_df.loc[:, 'email_address'] == 'NaN'
n_email_people_df = data_df.loc[n_email_people ] 
n_email_people =n_email_people_df.loc[:, ('poi','total_payments','total_stock_value',
                                          'email_address', 'from_messages', 'to_messages' )]
n_email_people.head(len(n_email_people))


# We have got some interesting information: there is someone called 'THE TRAVEL AGENCY IN THE PARK' and we have 'TOTAL'
# Lets see 'THE TRAVEL AGENCY IN THE PARK'

# In[24]:


data_dict['THE TRAVEL AGENCY IN THE PARK']


# We see some payments has been done to this 'THE TRAVEL AGENCY IN THE PARK', lets see in the net what is this.
# Interesting, this is a kind of travel agency half owned by the sister of... Kenneth. "While Enron's employees were in theory permitted to book their business travel elsewhere, Sharon Lay's was the agency of choice" By Jyoti Thottam/Houston write
# 
# It will be removed later in the process as well as 'Total'
# 
# How many Poi do we have in the data set?

# In[31]:


len(df_poi)


# In[ ]:


Let see who are the those 18 Poi (Person of interest), their total payment, stock and emails exchange


# In[55]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poi =poi_df.loc[:, ('poi','total_payments','total_stock_value',
                     'email_address', 'from_messages', 'to_messages' )]
poi.head(len(poi))


# We have 18 person of interes with their mails, but 4 has NaN in from poi and to poi messagess 

# Lets check the people name in the entries are correct by checking if they are surname name initial as we see in the head of the data set. This will allow us to find further divergency not found until now.

# In[23]:


import re
for index in data_df.index:
    if re.match('^[A-Z]+\s[A-Z]+(\s[A-Z])?$', index):
        continue
    else:
        print(index)


# Seams those names are right, we have some junior and some composed names.

# In[ ]:


https://nbviewer.jupyter.org/github/vsvasconcelos/ML/blob/master/DocumentoProjeto.html


# In[ ]:


data_dict_df_POI = data_dict_df[data_dict_df.poi==True]
data_dict_df_NPOI = data_dict_df[data_dict_df.poi==False]
mean_POI = pd.DataFrame(data_dict_df_POI.mean(), columns = ['POI'])
mean_NPOI = pd.DataFrame(data_dict_df_NPOI.mean(), columns = ['NPOI'])
df_mean = pd.concat([mean_POI, mean_NPOI], axis=1)
df_mean['POI_NPOI'] = df_mean['POI']/df_mean['NPOI']
df_mean.sort_values(by = "POI_NPOI", ascending=False)


# In[ ]:


Later we will check with PCA if the machine agree with us 

Now lets pass to the next point: data check


# In[ ]:




