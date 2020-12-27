#!/usr/bin/env python
# coding: utf-8

# # 1. DATA ADQUISITION 
# First thing to do is to create a dictionary with the features we need and we check the keys and the values of one key.

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

import warnings
warnings.filterwarnings('ignore')

### Loading the dictionary containing the dataset

data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

### running the function to remove the non person entries and correct the wrong values
### 'THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 'BELFER ROBERT', 'BHATNAGAR SANJAY'

my_dataset = data_dict


# Defining the list of features to be loaded

features_list=[                'salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi','email_address', 'poi']
        
# Creating the data frame from the dictionary with the features list defined and replacing Nan by 0

data_df = pd.DataFrame.from_dict(data_dict, orient='index')

data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df= data_df.loc[ :,(features_list )]  

data_df= data_df.replace('NaN', 0.0)
round(data_df.describe(),0)

data_df_0=data_df


# # 3. Features: analyse, create and chose features

# ## 3.1. Features correlation overview
# Lets start with a global view for all features in a color map showing their correlation (Persons)

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(16,12)})
sns.set(font_scale=0.7 )

sns.heatmap(data_df[[          'salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi','email_address', 'poi'
                     ]].corr(method="spearman"), 
                    cmap="RdYlBu", annot=True, fmt=".2f").set_title("Pearson Correlation Heatmap")

plt.show()


# We can see in the last line, poi, that there are not a single feauture that have more than 50% correlation with poi
# Lets see all features for non poi in color map

# In[4]:


# data_npoi: data split for 'poi'=False
data_npoi = data_df[data_df['poi'] ==False]

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")
corr = data_npoi.corr() * 100

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 11))

cmap = sns.diverging_palette(55, 100)

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            linewidths=1, cbar_kws={"shrink": .7}, annot=True, fmt=".2f")


# Beautiful isnt it? 
# Lets see now for poi 

# In[5]:


# data_poi: data split for 'poi'=True

data_poi = data_df[data_df['poi'] ==True]

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")
corr = data_poi.corr() * 100

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 11))

cmap = sns.diverging_palette(10, 650)

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            linewidths=1, cbar_kws={"shrink": .7}, annot=True, fmt=".2f")


# In[6]:


corr = data_df.corr(method="spearman") * 100
corr =corr.loc[ :,( 'poi', 'total_payments', 'to_messages')]
corr= round((corr.sort_values('poi',ascending=False)),0)
corr

As a first point we could see that poi have not director fees, neither restrictes stock deferred. 
We can see that salary-bonus combination is more important for poi than for non poi, then it could be a combine new feature.
Same happen with total stock value and total payment 
Regarding mails we see a difference between from this person to poi when is a poi.
# ## 3.2. Features detailed view with and without outliers
Lets see now the features by set of 5 features compare one by one, with poi identification in orange. We will take the ones with bigger correlation in the heatcharts
We should compare: bonus, salary, long_term_incentive, expenses and other
# In[7]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(data_df , vars=['salary', 'bonus', 'long_term_incentive','expenses','other'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# As we have some outlier in the diferent features, the visual is not good enought. Lets remove for each feature the max value we see in the data (using data_df.describe()) 

# In[8]:


round(data_df.describe( ),0)


# In[9]:


data_wo_outbon = data_df[data_df['salary'] <=1000000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['bonus'] <= 7500000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['long_term_incentive'] <= 5000000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['expenses'] <= 200000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['other'] <= 9500000]

g = sns.pairplot(data_wo_outbon , vars=['salary', 'bonus', 'long_term_incentive','expenses','other'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# Despite the previous outlier removal, it seams there is still one outlier in other and it is not a Poi. 

# In[10]:


d_other= data_wo_outbon.sort_values('other',ascending=False)
d_other.head(2)


# We see that Mark Frevert got more than 7m in other payment. 
# 
# Other payment reflects items such as payments for severence, consulting services, relocation costs, tax advances and allowances for employees on international assignment (i.e. housing allowances, cost of living allowances, payments under Enron’s Tax Equalization Program, etc.). May also include payments provided with respect to employment agreements, as well as imputed income amounts for such things as use of corporate aircraft.
# 
# And then if we search a bit on the net we found that Mr. Frevert was located in London as President
# and CEO of Enron Europe.
# 
# Then that explain why he has those level of others payment. 
# 
# Let see stock features. 

# In[11]:


g = sns.pairplot(data_df, vars=['exercised_stock_options', 'restricted_stock', 'total_stock_value'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# As we have some outlier in the diferent features, the visual is not good enought. Lets remove for each feature the max value we see in the data

# In[12]:


data_wo_outbon = data_df[data_df['total_stock_value'] <=49000000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['exercised_stock_options'] <=30000000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['restricted_stock'] <= 14000000]

g = sns.pairplot(data_wo_outbon, vars=['exercised_stock_options', 'restricted_stock', 'total_stock_value'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# Again we have difficulties to see a clear difference when seeing the features two by two.
# Let see the mails exchanges pair relations

# In[13]:


g = sns.pairplot(data_df , vars=[ 'from_messages', 'to_messages',  'shared_receipt_with_poi',
                               'from_poi_to_this_person', 'from_this_person_to_poi'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# There are some outlier in the emails too.
# 
# 

# ## 3.3. Outlier champions
# 
# I would like to check who are the outlier in the different features in each group, poi and non poi.

# In[14]:


#data_poi

IQR = data_poi.quantile(q=0.75) - data_poi.quantile(q=0.25)
first_quartile = data_poi.quantile(q=0.25)
third_quartile = data_poi.quantile(q=0.75)
outliers = data_poi[(data_poi>(third_quartile + 1.5*IQR) ) | (data_poi<(first_quartile - 1.5*IQR) )].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)
outliers.head()


# We have in our list of poi Lay Kenneth with 9 outlier
# But we have other people:
# 
# Timothy Belden is the former head of trading in Enron Energy Services. He is considered the mastermind of Enron's scheme to drive up California's energy prices, by developing many of the trading strategies that resulted in the California electricity crisis. Belden pleaded guilty to one count of conspiracy to commit wire fraud as part of a plea bargain, along with his cooperation with authorities to help convict many top Enron executives.
# 
# Jeffrey Keith Skilling is an American former businessman and convicted felon best known as the CEO of Enron Corporation during the Enron scandal. In 2006, he was convicted of federal felony charges relating to Enron's collapse and eventually sentenced to 24 years in prison. As a consultant for McKinsey & Company, Skilling worked with Enron during 1987, helping the company create a forward market in natural gas. Skilling impressed Kenneth Lay in his capacity as a consultant, and was hired by Lay during 1990 as chairman and chief executive officer of Enron Finance Corp. In 1991, he became the chairman of Enron Gas Services Co., which was a result of the merger of Enron Gas Marketing and Enron Finance Corp. Skilling was named CEO and managing director of Enron Capital and Trade Resources, which was the subsidiary responsible for energy trading and marketing. He was promoted to president and chief operating officer of Enron during 1997, second only to Lay, while remaining the manager of Enron Capital and Trade Resources.
# 
# Lets check know the outlier in non poi

# In[15]:


#data_poi
#data_npoi

IQR = data_npoi.quantile(q=0.75) - data_npoi.quantile(q=0.25)
first_quartile = data_npoi.quantile(q=0.25)
third_quartile = data_npoi.quantile(q=0.75)
outliers = data_npoi[(data_npoi>(third_quartile + 1.5*IQR) ) | (data_npoi<(first_quartile - 1.5*IQR) )].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)
outliers.head()


# I am going to keep this outlier in the data set for the time being and once I will decide which feature remains, I will check if they are outlier on those features.
# 
# 

# ## 3.4. Features Poi/non poi shadows 
# 
# Lets see know the features two by two

# In[16]:


# Function to plot two features, with tendency and outlier cleaning. We should pass to the function 
# the maximun value for each feature, that makes the cleaning of the outlier.

def bicolor_plot(feature_x, feature_y, feature_xstr, feature_ystr,outl_x, outl_y):
    from matplotlib import pyplot
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot
    from io import StringIO
    
    sb.lmplot(x=feature_x, y= feature_y, hue='poi', data=data_df, palette='Set1',height=5,markers=['P','o'])
    pyplot.title(feature_xstr +'/' +feature_ystr, fontsize=14)
    pyplot.xlabel(feature_xstr, fontsize=16)
    pyplot.ylabel(feature_ystr, fontsize=16)
    pyplot.show()
    
    data_wo_outbon_x = data_df[data_df[feature_x] <= outl_x]
    data_wo_outbon_y =  data_wo_outbon_x [ data_wo_outbon_x [feature_y] <= outl_y]

    sb.lmplot(x=feature_x, y= feature_y, hue='poi', data=data_wo_outbon_y, palette='bright',height=5,markers=['P','o'])
    pyplot.title(feature_xstr +'/' +feature_ystr, fontsize=14)
    pyplot.xlabel(feature_xstr, fontsize=16)
    pyplot.ylabel(feature_ystr, fontsize=16)
    pyplot.show()


# Let see mails from poi and to poi for poi and non poi. As we know the level of outlier, we could remove from the begining

# In[17]:


bicolor_plot("from_this_person_to_poi", "from_poi_to_this_person", "Mail to Poi", "Mail from Poi",150,200)


# In[18]:


bicolor_plot("long_term_incentive", "bonus", "long_term_incentive", "bonus",4000000,5000000)


# We see in these charts for bonus and long term incentive the effect of the outliers. When we use the complete set we have two divergents line, but when we remove several outliers, we get two paralel lines

# In[19]:


bicolor_plot("total_payments", "total_stock_value", "total_payments", "total_stock_value",17000000,48000000)


# ## 3.5. Features creation and overview of all features
# Next step is to create some features. There are several ways of create features, that could be a combination of existing features by direct mathematical operations or using PCA. We have created different features using those both methods. 

# We define one function to create some new features by mathematical operations and integrate them in the data set

# In[20]:


#Function to define new features
def create_df (features_original_list,features_amendm_list):
    
    
    import pickle
    from feature_format import featureFormat, targetFeatureSplit
    #from tester import dump_classifier_and_data
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import warnings
    warnings.filterwarnings('ignore')
    
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
              
             
    features_original_list.append("incentives") 
    features_original_list.append("incentives_ratio")
    features_original_list.append("total_money")
    
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

# In[21]:


# 1. original features
features_list0=[ 'poi',
                               'salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi','shared_receipt_with_poi']

# 2. features including new ones
features_list=[ 'poi',
                 'salary', 'bonus', 'long_term_incentive',
                 'deferred_income','deferral_payments', 'loan_advances',
                 'other', 'expenses','director_fees',  'total_payments',
                 'exercised_stock_options', 'restricted_stock',
                 'restricted_stock_deferred', 'total_stock_value',
                 'from_messages', 'to_messages', 
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
               
               "salary_ratio", "bonus_ratio","expenses_ratio", "other_ratio",
               "exercised_stock_options_ratio", "total_payments_ratio", "total_stock_value_ratio",
               "incentives_ratio",
               "from_poi_to_this_person_ratio", "from_this_person_to_poi_ratio",
               "incentives","total_money"   
             ]

# 3. run the function 
data_df=create_df (features_list0,features_list)

data_df


# We create some other features with PCA and integrate them in the data set.
# 
# payment_f=['salary', 'bonus', 'long_term_incentive', 'other', 'expenses']
# 
# payment_2=['salary', 'other', 'expenses']
# 
# payment_tt=['salary', 'bonus', 'long_term_incentive', 'deferred_income','deferral_payments', 'loan_advances',
#              'other', 'expenses','director_fees']
#              
# incentive_f=['bonus', 'long_term_incentive','exercised_stock_options']
# 
# stock_features=['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
# 
# emails_exc=['from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
#                 'shared_receipt_with_poi']

# In[22]:


# Features creation with PCA
        
payment_f=['salary', 'bonus', 'long_term_incentive', 'other', 'expenses']
payment_2=['salary', 'other', 'expenses']
payment_tt=['salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees']
incentive_f=['bonus', 'long_term_incentive','exercised_stock_options']

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


data_df  = data_df .loc[ :,(  'poi',
                             'salary', 'bonus', 'long_term_incentive',
                             'deferred_income','deferral_payments', 'loan_advances',
                             'other', 'expenses','director_fees',  'total_payments',
                             'exercised_stock_options', 'restricted_stock',
                             'restricted_stock_deferred', 'total_stock_value',
                             'from_messages', 'to_messages', 
                             'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
               
                           "salary_ratio", "bonus_ratio","expenses_ratio", "other_ratio",
                           "exercised_stock_options_ratio", "total_payments_ratio", "total_stock_value_ratio",
                           "incentives_ratio",
                           "from_poi_to_this_person_ratio", "from_this_person_to_poi_ratio",
                           "incentives","total_money",
                            
                            'incentive_f','payment_f','payment_2','payment_tt','stock_features','emails_exc')]

data_df=round(data_df,2)
data_df.head()



# We need to specify the origina feature list and the final one to run the function 

# In[23]:


features_list=[ 'poi',
                             'salary',"salary_ratio",
                             'bonus',  "bonus_ratio",
                             'long_term_incentive',
                             'deferred_income','deferral_payments', 'loan_advances',
                             'other',"other_ratio", 'expenses',"expenses_ratio",
                             'director_fees',  'total_payments',"total_payments_ratio",
                            
                             'exercised_stock_options',  "exercised_stock_options_ratio",'restricted_stock',
                             'restricted_stock_deferred', 'total_stock_value',"total_stock_value_ratio",
                             
                               "incentives","incentives_ratio", 'incentive_f',"total_money",
                              'payment_f','payment_2','payment_tt',
                     
                             'from_messages', 'to_messages', 
                             'from_poi_to_this_person',  "from_poi_to_this_person_ratio",
                             'from_this_person_to_poi',  "from_this_person_to_poi_ratio",
                             'shared_receipt_with_poi','emails_exc']




data_dict=data_df.to_dict('index')
my_dataset=data_dict
my_dataset


data_df = pd.DataFrame.from_dict(my_dataset, orient='index')

data_df = pd.DataFrame.from_dict(my_dataset, orient='index')
data_df= data_df.loc[ :,(features_list )]  

data_df= data_df.replace('NaN', 0.0)

data_df

round(data_df.describe(),2)


# Lets see all the features heatmap

# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(16,12)})
sns.set(font_scale=0.7 )

sns.heatmap(data_df[[  'poi',
                             'salary',"salary_ratio",
                             'bonus',  "bonus_ratio",
                             'long_term_incentive',
                             'deferred_income','deferral_payments', 'loan_advances',
                             'other',"other_ratio", 'expenses',"expenses_ratio",
                             'director_fees',  'total_payments',"total_payments_ratio",
                            
                             'exercised_stock_options',  "exercised_stock_options_ratio",'restricted_stock',
                             'restricted_stock_deferred', 'total_stock_value',"total_stock_value_ratio",
                             
                               "incentives","incentives_ratio", 'incentive_f',"total_money",
                              'payment_f','payment_2','payment_tt',
                     
                             'from_messages', 'to_messages', 
                             'from_poi_to_this_person',  "from_poi_to_this_person_ratio",
                             'from_this_person_to_poi',  "from_this_person_to_poi_ratio",
                             'shared_receipt_with_poi','emails_exc'
                     ]].corr(method="spearman"), 
                    cmap="RdYlBu", annot=True, fmt=".2f").set_title("Pearson Correlation Heatmap")

plt.show()


# Lets see the correlations of poi with the different features by descending order

# In[25]:


corr = data_df.corr(method="spearman") * 100
corr =corr.loc[ :,( 'poi', "incentives","incentives_ratio", 'incentive_f')]
corr= round((corr.sort_values('poi',ascending=False)),0)
corr


# We see that there are some features that are correlated among them, we could remove them, eg payment_f, payment_2 and payment_tt, we keep payment_f as  isthe one with more correlation with poi.
# Lets see some pairplot with new features

# # 3.6. Robust Scaler
# We have investigate different scaler, they are visible in the files 3.a. and 3.b. 
# We dedice to use Robust Scaler and is the one we aply here 
# Lets create the different list of features that we will use in the different steps

# # 3.6.1 Features lists

# In[26]:


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

# 9. orgianl features list without poi . The one to pass to the algo
algo_list0= list(set(features_list0).difference(set(poi)))
algo_list0= list(set(algo_list0).difference(set(features_low)))


# # 3.6.2. Scaler
# 
# Before scaling we are going to remove the features with more than 70% of Nan in the person that are Poi and at total data set:  'director_fees' (100%, 90%), 'restricted_stock_deferred' (100%, 88%), 'loan_advances' (94%, 98%), ‘deferral_payments' (72%, 74%) 
# Same for 'deferred_income' because that is a negative value that represent a discount in the payment and it is missing in 66% of the records
# 
# Lets create the different list of features that we will use in the different steps
# 

# In[27]:


#Remove of features higher Nan
data_df= data_df.loc[ :,(features_list_cl)] 


# In[28]:


round(data_df.describe(),2)

data_des_t=(round(data_df.describe(),0)).transpose()
data_des_t= round((data_des_t.sort_values('max',ascending=False)),0)
data_des_t


# We use a funtion for scale with RobustScaler and we run the function and the description of the dataframe after scaling

# In[29]:


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

data_rob_des_t=(round(data_df_rob.describe(),0)).transpose()
data_rob_des_t= round((data_rob_des_t.sort_values('max',ascending=False)),0)
data_rob_des_t


# Let see the correlation of some of the features after scalation

# In[30]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(data_df_rob , vars=['salary', 'bonus', 'long_term_incentive','expenses','other'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# The correlation among the features does not change after scaling. The relations between the features and the poi/non poi is not significant for any features that means the algorithm is going to struggle to find the right solution. 
# As there are more non poi that poi, most probably we will get good ratios for prediction non poi than poi

# # 3.6.3 Create file (pkl) with scaled features

# In[31]:


data_dict_rob=data_df_rob.to_dict('index')
my_dataset_rob=data_dict_rob
import pickle
pickle.dump(my_dataset_rob, open('data_dict_rob.pkl', "wb"))

