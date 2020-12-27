#!/usr/bin/env python
# coding: utf-8

# # Enron mails vs poi - Machine learning project

# 1. Data adquisition: import data, first view of data and curious findings
# 2. Data check: cleaning NaN, visualize data and check outliers
# 3. Features: analyse, create and chose features
# 4. Try clasifiers and check scores
# 5. Chose one clasifier and tune it

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
# ## 1.1. Dataframe creation
# First thing to do is to create a dictionary with the features we need, convert that dictionary in a data frame and we check the keys and the values of one key.

# In[1]:


### Import some packages and modules to be used later
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

### Loading the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

# Creating the data frame from the dictionary with the existing features

data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df  = data_df .loc[ :,('poi','salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments', 'restricted_stock_deferred','exercised_stock_options', 
                 'restricted_stock', 'total_stock_value', 'email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]
data_df.head()


# It seams we have several 'NaN' *no information avalaible*. We need to clean those 'NaN' before start to analyse our data
# We can see that some people has not email address, but have salary and other financial figures. Lets keep that in mind for later investigation (eg. how many people without adrress mail do we have? have they any specific commmon point? is that an issue when compiling data?)

# ## 1.2. Dataframe overview
# 
# Let's see a bit the structure, type and content of our data

# In[2]:


data_df.info()


# We have 146 people in our data with 21 posibles informations but we do not know what is the 'NaN' volume per person and per information. We need to be aware of those 'NaN' to be able to correctly interpretate the posible output of our investigation

# Lets see now the same information for the Poi and non poi entries

# In[3]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]
poi_df.info()


# In[4]:


non_poi = data_df.loc[:, 'poi'] == False
non_poi_df = data_df.loc[non_poi]
non_poi_df.info()


# Let take a look on the Nan by person and feature in the dataset

# In[5]:


data_df1= data_df.replace('NaN', 0.0)
data_df1  = data_df1.replace(0, np.nan)
data_df1.isna()
data_df1t=data_df1
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(15,25))
sns.heatmap(data_df1t.isna(),yticklabels=True,cbar=False)


# The volume of NaN is different for each feature and person, but we can see some person and some features with nearly no data
# The point is that for financial features the NaN means that the person have got 0 on that feature as we can check in pdf document (enron61702insiderpay.pdf). 
# 
# Lets summarize Nan per feature

# In[6]:


nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
round((nan_summary.sort_values('nan-proportion',ascending=False)),2)


# The level on Nan is high for some features. 
# It seams logic to remove all the features with more than 70% of Nan in the person that are Poi and at total data set :  'director_fees' (0%, 10%), 'restricted_stock_deferred' (0%, 89%), 'loan_advances' (93%, 98%), ‘deferral_payments' (71%, 73%) 
# Same for 'deferred_income' because that is a negative value that represent a discount in the payment and it is missing in 65% of the records
# Regarding the emails, this project use the dataset corresponding to the counting of mails and not gone to the details of the text in the emails. Taking into account that the total amount of emails for each person has been cleaned in a more or less proper way, seams correct not to use that total amount but only the mails with relation with poi, that means send to or received from poi. On top, there are 40% of persons without mails

# ## 1.3. NaN poi and non-poi 
# Lets see if big differences on poi and nonpoi regarding the level of Nan in features

# In[7]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poi =poi_df.loc[:, ('poi','salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments', 'restricted_stock_deferred','exercised_stock_options', 
                 'restricted_stock', 'total_stock_value', 'email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]

non_poi = data_df.loc[:, 'poi'] == False
non_poi_df = data_df.loc[non_poi]

non_poi =non_poi_df.loc[:, ('poi','salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments', 'restricted_stock_deferred','exercised_stock_options', 
                 'restricted_stock', 'total_stock_value', 'email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]


# In[8]:


pnan_summary = pd.DataFrame({'size': poi.count(),
                            'no-nan': poi.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            poi.columns)
pnan_summary['nan-proportion'] =1-( pnan_summary['no-nan'] / pnan_summary['size'])
round((pnan_summary.sort_values('nan-proportion',ascending=False)),2)


# We have 18 person of interes their mails, but 4 has NaN in from poi and to poi messages.
# Regarding financial features, Nan is not a real missing information but that for that category there was no payment or stock value
# Nan in Poi are mainly in director fees, restricted_stock_deferred and loan advances

# In[9]:


pnan_summary = pd.DataFrame({'size': non_poi.count(),
                            'no-nan': non_poi.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            non_poi.columns)
pnan_summary['nan-proportion'] =1-( pnan_summary['no-nan'] / pnan_summary['size'])
round((pnan_summary.sort_values('nan-proportion',ascending=False)),2)


# We have 128 person of interes thereoff 93 have email address but we have recorded mails only for 72
# Nan in Poi are mainly in director fees, restricted_stock_deferred and loan advances

# ## 1.4. Nan in Total Payment / Total Stock
# Lets see a bit what is behind the entries where total payment is NaN. 

# In[10]:


t_payms = data_df.loc[:, 'total_payments'] == 'NaN'
df_t_payms = data_df.loc[t_payms]

df_t_payms=df_t_payms.replace('NaN', 0.0)

df_t_payms = df_t_payms.loc[:, ('poi','total_payments','total_stock_value',
                           'salary', 'bonus',  'expenses', 'other', 'long_term_incentive',
                           'exercised_stock_options', 'restricted_stock',
                           'email_address', 'from_messages', 'to_messages', 
                           'shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi')]

df_t_payms = df_t_payms.sort_values(['total_payments','total_stock_value'], ascending=[True,True])

df_t_payms.head(len(df_t_payms))


# There are 3 entries with no payment, no stock: CHAN RONNIE, POWERS WILLIAM and LOCKHART EUGENE E.
# Lets investigate them a bit

# In[11]:


data_dict['CHAN RONNIE']


# Ronnie Chan: NaN at total payment and total stock comes from the fact that director fees and restricted stock are neutralized by deferred income and restricted stock deferred.
# 
# Note: Ronnie Chan Chi-chung is a Hong Kong billionaire businessman. He was a director of Enron Corporation and a member of its audit committee when it filed for bankruptcy as a result of fraud.
# 
# We will see later when we will reduce the number of features if this entry reamains in the data set

# In[12]:


data_dict['POWERS WILLIAM']


# William Powers: NaN in this case should be 0 as the director fees are neutralized by deferred income. 
# Note: Powers — a lawyer and professor — was tapped by Enron’s board to lead a special investigation to determine what went wrong and why. He has a enron mail but no exchange mails with Poi.

# In[13]:


data_dict['LOCKHART EUGENE E']


# LOCKHART EUGENE E with only one information: he is not poi. We could consider the added value is to inform the algorithm that entries with no values are non-poi. This entry could be deleted but before lets see who is this person.
# 
# Eugene Lockhart whas the CEO of a company created in May 2020: 
# "Based in Greenwich, Conn., with some operations in Houston, The New Power Company will be headed by President and CEO H. Eugene Lockhart. Lockhart was formerly president of AT&T (T: Research, Estimates) consumer services, president of Bank America's Global Retail Bank, and CEO of MasterCard International."
# And what is more curious is what offer this company
# "Through our agreement with the New Power Company, an innovator in the field of energy services, online consumers for the first time can simply and efficiently find out more information about alternative energy sources, compare and select new energy service providers, and receive and pay their utility bills online," said Bob Pittman, president and COO of AOL Inc."
# 
# https://money.cnn.com/2000/05/16/technology/enron/
# 
# The other people with no payment from Enron have stock values or enron emails and have exchange emails with poi.
# We will keep them for the time being.

# ## 1.5. Nan in email address
# Regarding the mails, we see that there are 111 entries with email address but only 86 entries with values on the messages categories. Let’s see who the people without email is, check that the entries with no email adress does no have values in emails data and if they have values in payments and stock

# In[14]:


n_email_people = data_df.loc[:, 'email_address'] == 'NaN'
n_email_people_df = data_df.loc[n_email_people ] 
n_email_people =n_email_people_df.loc[:, ('email_address', 'from_messages', 'to_messages', 
                                          'shared_receipt_with_poi',
                                          'from_poi_to_this_person', 'from_this_person_to_poi',
                                          'total_payments','total_stock_value',
                                          'poi' )]

n_email_people= n_email_people.replace('NaN', 0.0)

#n_email_people = n_email_people.sort_values('total_payments',ascending=True)
n_email_people = n_email_people.sort_values(['total_payments','total_stock_value'], ascending=[True,True])
#df.sort_values(['b', 'c'], ascending=[True, False], inplace=True)

n_email_people.head(len(n_email_people))


# There are two entries with no emails that we have check in the previous point.
# There are several entries with no email but with financial data.
# We found two entries that could be mistakes: one of the entries without emails is TOTAL that is the sum of the financial data per entrie and the other one is 'THE TRAVEL AGENCY IN THE PARK'. 
# Lets investigate them

# ## 1.6. Not valid entries
# We have got some interesting information: there is someone called 'THE TRAVEL AGENCY IN THE PARK' and we have 'TOTAL'.
# Lets see 'THE TRAVEL AGENCY IN THE PARK'

# In[15]:


data_dict['THE TRAVEL AGENCY IN THE PARK']


# We see some payments has been done to this 'THE TRAVEL AGENCY IN THE PARK', lets see in the net what is this.
# Interesting, this is a kind of travel agency half owned by the sister of... Kenneth. "While Enron's employees were in theory permitted to book their business travel elsewhere, Sharon Lay's was the agency of choice" By Jyoti Thottam/Houston write
# 
# It will be removed later in the process as well as 'Total'

# Lets check the people name in the entries are correct by checking if they are surname name initial as we see in the head of the data set. This will allow us to find further divergency not found until now.

# In[16]:


import re
for index in data_df.index:
    if re.match('^[A-Z]+\s[A-Z]+(\s[A-Z])?$', index):
        continue
    else:
        print(index)


# Seams those names are right, we have some junior and some composed names.

# ## 1.7. Data checking and correction
# 
# Lets make a first old school check to ensure that total payment and total stock value are the sum of the different payments and stock features respectively

# In[17]:


# Define what is payment data 
payment_data=['salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees','total_payments']

# Create data frame of payment with payment data for total payment not NaN. Replace NaN by 0.0. 

payms = data_df.loc[:, 'total_payments']  != 'NaN'
payms_df = data_df.loc[payms]

payms = payms_df.loc[ :,('salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments')]

payms = payms.replace('NaN', 0.0)

payms_df = payms_df.replace('NaN', 0.0)

t_pay_wrong = (payms [payms [payment_data[:-1]].sum(axis='columns') != payms_df ['total_payments']])

t_pay_wrong 


# There are two lines where the sum of payments are not equal to the total payments

# In[18]:


# Define what is stock data
stock_data=['restricted_stock_deferred', 'exercised_stock_options', 'restricted_stock', 'total_stock_value' ]

# Create data frame of stocks with stock data for total stock not NaN. Replace NaN by 0.0. 

stocks = data_df.loc[:, 'total_stock_value']  != 'NaN'
stocks_df = data_df.loc[stocks]

stocks = stocks_df.loc[ :,('restricted_stock_deferred','exercised_stock_options', 
                           'restricted_stock', 'total_stock_value')]

stocks = stocks.replace('NaN', 0.0)

stocks_df = stocks_df.replace('NaN', 0.0)

t_stock_wrong = (stocks [stocks [stock_data[:-1]].sum(axis='columns') != stocks_df ['total_stock_value']])

t_stock_wrong 


# There is only one register where the sum of stocks informations are not equal to the total stock value. And that is for the same person, Belfer Robert. Lets go to the pdf with the insider payment to check where the issue comes from.

# In[19]:


data_dict['BELFER ROBERT']


# In fact, none of the informations about payments or stock are right foer R. Belfer.
# But who is R. Belfer and what was his role in the colapse?
# Belfer was chairman of Belco Petroleum Corp. that was merged into one of the predecessors of Enron Corp, the Omaha, Nebraska-based InterNorth, Inc., and the Belfer family received a sizeable equity stake in the transaction, eventually becoming Enron's largest shareholder. Belfer served the board of directors of Enron and was estimated to have held over 16 million Enron shares as of August, 2000. However, he was reported to be reticent on the board and was not involved in the operations of the company. He resigned from the board in June, 2002.
# 
# We have two options, we delete his register from the data set or we correct them

# In[20]:


data_dict['BHATNAGAR SANJAY']


# In fact, none of the informations about payments are right, and we see that stock options neither.
# But who is S. Bhatnagar and what was his role in the colapse?
# Mr. Bhatnagar has served as the chairman and CEO of Enron South Asia where his responsibilities included developing and financing energy infrastructure. He was the lead developer of the 2,184 megawatt Dabhol Power project in India, which began commercial operations in May 1999.
# 
# We have two options, we delete his register from the data set or we correct them
# The information related to mails is that he send 29 mails and only 1 to a poi. 

# As we do not have too much entries we are going to correct them.
# First we create the entries with the correct data, we pass to the dataset and we check again if there is any mistake in the data

# In[39]:


# We create a function to correct the mistake and to delete the not valid entries

# We create a function to do that as we need to do every time we load the dataser

def data_dict_cor (data_dict):
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
    return data_dict


# In[22]:


data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

data_dict_cor (data_dict)

data_dict['BELFER ROBERT']


# In[23]:


data_dict['BHATNAGAR SANJAY']


# In[24]:


data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df  = data_df .loc[ :,('poi','salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments', 'restricted_stock_deferred','exercised_stock_options', 
                 'restricted_stock', 'total_stock_value', 'email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]


# In[25]:


# Define what is payment data 
payment_data=['salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees','total_payments']

# Create data frame of payment with payment data for total payment not NaN. Replace NaN by 0.0. 

payms = data_df.loc[:, 'total_payments']  != 'NaN'
payms_df = data_df.loc[payms]

payms = payms_df.loc[ :,('salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments')]

payms = payms.replace('NaN', 0.0)

payms_df = payms_df.replace('NaN', 0.0)

t_pay_wrong = (payms [payms [payment_data[:-1]].sum(axis='columns') != payms_df ['total_payments']])

t_pay_wrong 


# Now no lines have delta betwen sum of payments and total payments

# In[26]:


# Define what is stock data
stock_data=['restricted_stock_deferred', 'exercised_stock_options', 'restricted_stock', 'total_stock_value' ]

# Create data frame of stocks with stock data for total stock not NaN. Replace NaN by 0.0. 

stocks = data_df.loc[:, 'total_stock_value']  != 'NaN'
stocks_df = data_df.loc[stocks]

stocks = stocks_df.loc[ :,('restricted_stock_deferred','exercised_stock_options', 
                           'restricted_stock', 'total_stock_value')]

stocks = stocks.replace('NaN', 0.0)

stocks_df = stocks_df.replace('NaN', 0.0)

t_stock_wrong = (stocks [stocks [stock_data[:-1]].sum(axis='columns') != stocks_df ['total_stock_value']])

t_stock_wrong 


# Now no lines have delta betwen sum of stock and total stock

# We can continue with the analysis of the Nan in the dataset

# ## 1.8. Top Nan entries
# Lets now check the levels of Nan per person

# In[27]:


data_df1= data_df
data_df1t=data_df1.T
        
nan_summary = pd.DataFrame({'size': data_df1t.count(),
                            'no-nan': data_df1t.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df1t.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
data_nan=(round((nan_summary.sort_values('nan-proportion',ascending=False)),2))
data_nan.head(30)


# There are several entries with nearly no information. We have identified until now some of the entries in this table, lets see some of them

# In[28]:


data_dict['GRAMM WENDY L']


# This is one of the entry we see in director fees. It is not a Poi and have no information related to emails, 

# In[29]:


data_dict['WHALEY DAVID A']


# This entry have only stock information, it is not a Poi and have no information related to emails, 

# In[30]:


data_dict['WROBEL BRUCE']


# This entry have low value in director fees and stock value only stock information, it is not a Poi and have no information related to emails.

# In[31]:


data_dict['WAKEHAM JOHN']


# WAKEHAM JOHN was Enron board memeber. This entry have low value in director fees and expenses, it is not a Poi and have no information related to emails.
# He got payments on expenses as he was living abroad (UK) and got expenses when traveling for Board meetings, then we keep him in our data set as this feature is used.
# 
# "Three of the six departing directors live overseas: Ronnie C. Chan, 52, chairman of Hong Kong real estate firm Hang Lung Group; Paulo V. Ferraz Pereira, 47, executive vice president of Brazil’s Grupo Bozano banking firm; and Lord John Wakeham, 69, former secretary of state for energy in Britain."
# https://www.latimes.com/archives/la-xpm-2002-feb-13-fi--board13-story.html

# ## 1.9. Nan per entry: who is who
# 
# We see a lot of Nan in the data set. Lets go deeper to see who are the entries with big level of Nan but for the features where we have less Nan and are not sum of other features

# In[41]:


# Create a data frame to see the level of Nan per entrie


data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
   
data_dict_cor (data_dict)

my_dataset = data_dict

data_df = pd.DataFrame.from_dict(data_dict, orient='index')

features_list=['salary', 'bonus', 'long_term_incentive',
                        # 'deferred_income','deferral_payments', 'loan_advances','director_fees',
                'other', 'expenses',  
                     # 'total_payments',
                 'exercised_stock_options', 'restricted_stock',
                        #  'restricted_stock_deferred', 'total_stock_value',
                 'from_messages', 'to_messages', 
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                'shared_receipt_with_poi','email_address', 'poi']

data_df1= data_df.loc[ :,(features_list )]   

data_df1t=data_df1.T
        
nan_summary = pd.DataFrame({'size': data_df1t.count(),
                            'no-nan': data_df1t.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df1t.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
data_nan=(round((nan_summary.sort_values('nan-proportion',ascending=False)),2))
data_nan.head(25)


# Lets see who of them got payment from director fees 

# In[33]:


data_df  = data_df .loc[ :,(  'poi','director_fees','total_payments', 'total_stock_value',
                             'salary', 'bonus', 'long_term_incentive',
                             'deferred_income','deferral_payments', 'loan_advances',
                             'other', 'expenses',  
                             'exercised_stock_options', 'restricted_stock',
                             'restricted_stock_deferred',
                             'from_messages', 'to_messages', 
                             'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
               )]

dir_fees = data_df.replace('NaN', 0.0)
dir_fees = dir_fees.sort_values('director_fees',ascending=False)

dir_fees.head(14)


# We have identified until now some of the entries in this table, lets see the ones that have not stock payments
# 
# SAVAGE FRANK His payments comes from director fees but he have deferred those incomes
# 
# BLAKE JR. NORMAN His payments comes from director fees but he have deferred those incomes. He got payment in expenses
# 
# WINOKUR JR. HERBERT His payments comes from director fees but he have deferred those incomes. He got payment in expenses
# 
# MENDELSOHN JOHN His payments comes from director fees but he have deferred those incomes. He got payment in expenses
# 
# MEYER JEROME J His payments comes from director fees but he have deferred those incomes. He got payment in expenses
# 
# 
# And the reason why the entries with payment on director fees has amounts on deferred payment and stock is behind a company requirement
# https://www.latimes.com/archives/la-xpm-2002-feb-13-fi--board13-story.html
# "Non-employee Enron directors each have been paid $50,000 in service fees annually, plus $1,250 for each board meeting attended. The directors were required to defer 50% of their base annual service fee into an Enron stock plan."

# In[34]:


data_dict['PEREIRA PAULO V. FERRAZ']


# PEREIRA PAULO V. FERRAZ His payments comes from director fees but he have deferred those incomes. He was Enron board memeber. He got payments on expenses as he was living abroad (Brasil) and got expenses when traveling for Board meetings, then we keep him in our data set as this feature is used.

# Lets see now who are some of the other top Nan entries

# In[35]:


data_dict['URQUHART JOHN A']


# URQUHART JOHN A payments comes from expenses, then we keep him in our data set as this feature is used.
# 
# He was Enron board memeber and got some payment for consulting services. "The report also says Enron's executives compromised the independence of some board members with consulting payments. Enron paid board member John Urquhart $493,914 for consulting in 2000. " https://www.theglobeandmail.com/report-on-business/us-senate-blasts-enron-directors/article25694090
# 

# In[36]:


data_dict['CLINE KENNETH W']


# CLINE KENNETH W we found a mention to this person in the "Official Summary of Security Transactions and Holdingsok". His role is not clear. The information avalaible is on the features we decided to keep then we keep him in our dataset

# In[ ]:


data_dict['WAKEHAM JOHN']


# WAKEHAM JOHN was Enron board memeber. He got payments on expenses as he was living abroad (UK) and got expenses when traveling for Board meetings, then we keep him in our data set as this feature is used.
# 
# "Three of the six departing directors live overseas: Ronnie C. Chan, 52, chairman of Hong Kong real estate firm Hang Lung Group; Paulo V. Ferraz Pereira, 47, executive vice president of Brazil’s Grupo Bozano banking firm; and Lord John Wakeham, 69, former secretary of state for energy in Britain."
# https://www.latimes.com/archives/la-xpm-2002-feb-13-fi--board13-story.html
# 

# In[37]:


data_dict['WODRASKA JOHN']


# WODRASKA JOHN his payments comes from other, then we keep him in our data set as this feature is used.
# 
# Woody Wodraska is a veteran of the water industry and started his professional career with the South Florida Water Management District (SFWMD). He was then recruited to serve as the CEO of the Metropolitan Water District of Southern California (MWD). Wodraska has worked in a variety of positions in the private sector of the water industry. This included Azurix, a subsidiary of ENRON and a stint with large to mid-sized Architectural & Engineering Consulting firms, specializing in water. Presently he is President of Wodraska Partners, a small firm specializing in providing comprehensive problem solving strategies for public and private entities in the water industry https://www.verdexchange.org/profile/john-woody-wodraska
# 

# In[38]:


data_dict['SCRIMSHAW MATTHEW']


# SCRIMSHAW MATTHEW was Enron . He got payments on expenses as he was living abroad (UK) and got expenses when traveling for Board meetings, then we keep him in our data set as this feature is used.
# 
# I found in the net a mail exchange with a poi
# http://www.enron-mail.com/email/skilling-j/sent/Re_UK_CEOs_6.html
# but that does not appear in our data set.
# That means again that the data quality of our data set is not 100% accurate, what is the normal situation when working with real data and human data cranching.

# In[ ]:


data_dict['WAKEHAM JOHN']


# We could remove all those entries for the data set arguing they do not give a lot of information but for the time being and knowing that we do not have a big amount of data, we keep all those entries as no error on them
