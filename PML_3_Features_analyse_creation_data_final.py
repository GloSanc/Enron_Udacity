#!/usr/bin/env python
# coding: utf-8

# In[60]:


import os 

os.chdir(r'C:\Users\sanchez_sanc\Desktop\data_analyst\Curso\4_Machine_learning\Curso_PY_3\tools') 

import sys
import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sys.path.append("../tools/")

import warnings
warnings.filterwarnings('ignore')

### Loading the dictionary containing the dataset
# the directory where you want to go
os.chdir(r'C:\Users\sanchez_sanc\Desktop\data_analyst\Curso\4_Machine_learning\Curso_PY_3\final_project') 
sys.path.append("../Curso_PY_3/final_project/")

list_out=['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E','GRAMM WENDY L',
          'WHALEY DAVID A','WROBEL BRUCE', 'BELFER ROBERT', 'BHATNAGAR SANJAY', 
              'KAMINSKI WINCENTY J','SHAPIRO RICHARD S','KEAN STEVEN J','FASTOW ANDREW S',
              'KOPPER MICHAEL J','YEAGER F SCOTT','HIRKO JOSEPH']

features_list=['salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi','email_address', 'poi'
              ]

data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
   
for name in list_out:
    data_dict.pop(name,0)
    
#data_dict['LAY KENNETH L']

my_dataset = data_dict
data_df = pd.DataFrame.from_dict(data_dict, orient='index')

data_df= data_df.loc[ :,(features_list )]  


# In[61]:


data_df= data_df.replace('NaN', 0.0)
round(data_df.describe(),0)

data_df_0=data_df
data_df_0


# # 3: Analyse existing features and create new ones#

# Lets start with a global view for all features in a color map showing their correlation (Persons)

# In[62]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(16,12)})
sns.set(font_scale=0.7 )

sns.heatmap(data_df[['salary', 'bonus', 'long_term_incentive',
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

# In[63]:


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

# In[64]:


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

As a first point we could see that poi have not director fees, neither restrictes stock deferred. 
We can see that salary-bonus combination is more important for poi than for non poi, then it could be a combine new feature.
Same happen with total stock value and total payment 
Regarding mails we see a difference between from this person to poi when is a poi.
Lets see now the features by set of 5 features compare one by one, with poi identification in orange. We will take the ones with bigger correlation in the heatcharts
We should compare: bonus, salary, long_term_incentive, expenses and other
# In[65]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(data_df , vars=['salary', 'bonus', 'long_term_incentive','expenses','other'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# As we have some outlier in the diferent features, the visual is not good enought. Lets remove for each feature the max value we see in the data (using data_df.describe()) 

# In[66]:


round(data_df.describe( ),0)


# In[67]:


data_wo_outbon = data_df[data_df['salary'] <=1000000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['bonus'] <= 7500000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['long_term_incentive'] <= 5000000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['expenses'] <= 200000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['other'] <= 9500000]

g = sns.pairplot(data_wo_outbon , vars=['salary', 'bonus', 'long_term_incentive','expenses','other'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# Despite the previous outlier removal, it seams there is still one outlier in other and it is not a Poi. 

# In[68]:


d_other= data_wo_outbon.sort_values('other',ascending=False)
d_other.head(1)


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

# In[69]:


g = sns.pairplot(data_df, vars=['exercised_stock_options', 'restricted_stock', 'total_stock_value'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# As we have some outlier in the diferent features, the visual is not good enought. Lets remove for each feature the max value we see in the data

# In[70]:


data_wo_outbon = data_df[data_df['total_stock_value'] <=49000000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['exercised_stock_options'] <=34000000]
data_wo_outbon = data_wo_outbon[ data_wo_outbon ['restricted_stock'] <= 14000000]

g = sns.pairplot(data_wo_outbon, vars=['exercised_stock_options', 'restricted_stock', 'total_stock_value'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# Again we have difficulties to see a clear difference when seeing the features two by two.
# Let see the mails exchanges pair relations

# In[71]:


g = sns.pairplot(data_df , vars=[ 'from_messages', 'to_messages',  'shared_receipt_with_poi',
                               'from_poi_to_this_person', 'from_this_person_to_poi'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# There are some outlier in the emails too.
# 
# I would like to check who are the outlier in the different features in each group, poi and non poi.

# In[72]:


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

# In[73]:


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
# Lets see know the features two by two

# In[74]:


# Function to plot two features, with tendency and outlier cleaning. We should pass to the function 
# the maximun value for each feature, that makes the cleaning of the outlier.

def bicolor_plot(feature_x, feature_y, feature_xstr, feature_ystr,outl_x, outl_y):
    from matplotlib import pyplot
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot
    from io import StringIO
    
    sb.lmplot(x=feature_x, y= feature_y, hue='poi', data=data_df, palette='Set1',height=4,markers=['P','o'])
    pyplot.title(feature_xstr +'/' +feature_ystr, fontsize=14)
    pyplot.xlabel(feature_xstr, fontsize=16)
    pyplot.ylabel(feature_ystr, fontsize=16)
    pyplot.show()
    
    data_wo_outbon_x = data_df[data_df[feature_x] <= outl_x]
    data_wo_outbon_y =  data_wo_outbon_x [ data_wo_outbon_x [feature_y] <= outl_y]

    sb.lmplot(x=feature_x, y= feature_y, hue='poi', data=data_wo_outbon_y, palette='bright',height=6,markers=['P','o'])
    pyplot.title(feature_xstr +'/' +feature_ystr, fontsize=14)
    pyplot.xlabel(feature_xstr, fontsize=16)
    pyplot.ylabel(feature_ystr, fontsize=16)
    pyplot.show()


# Let see mails from poi and to poi for poi and non poi. As we know the level of outlier, we could remove from the begining

# In[75]:


bicolor_plot("from_this_person_to_poi", "from_poi_to_this_person", "Mail to Poi", "Mail from Poi",150,200)


# In[76]:


bicolor_plot("long_term_incentive", "bonus", "long_term_incentive", "bonus",4000000,5000000)


# We see in these charts for bonus and long term incentive the effect of the outliers. When we use the complete set we have two divergents line, but when we remove several outliers, we get two paralel lines

# In[77]:


bicolor_plot("total_payments", "total_stock_value", "total_payments", "total_stock_value",17000000,48000000)


# After all this, I have my own idea of which features will stay in my dataset and the features I could create
# 
# The level on Nan is high for some features. I remove all the features with more than 70% of Nan in the person that are Poi and at total data set : 'director_fees' (0%, 10%), 'restricted_stock_deferred' (0%, 89%), 'loan_advances' (93%, 98%), ‘deferral_payments' (71%, 73%) I remove 'deferred_income' because that is a negative value that represent a discount in the payment and it is missing in 65% of the records Regarding the emails, this project use the dataset corresponding to the counting of mails and not gone to the details of the text in the emails. Taking into account that the total amount of emails for each person has been cleaned in a more or less proper way, I decided not to use that total amount but only the mails with relation with poi, that means send to or received from poi. On top, there are 40% of persons without mails
# 
# We create some features and we analyse the correlation with poi features. 
# New features 
# "incentives"  = bonus + long_term_incentive + exercised_stock_options
# 
# "total_money" = total_payments + total_stock_value
# 
# "bonus_ratio" = bonus / total_money
# 
# "incentives_ratio" = incentives /total_money 
# 

# In[78]:



for v in data_dict.values():
    
    bonus = v["bonus"]
    bonus=(float( bonus) if bonus not in ["NaN"] else 0.0)

    long_term_incentive = v["long_term_incentive"]
    long_term_incentive=(float( long_term_incentive) if long_term_incentive not in ["NaN"] else 0.0)
    
    exercised_stock_options =v["exercised_stock_options"] 
    exercised_stock_options=(float( exercised_stock_options) if exercised_stock_options not in ["NaN"] else 0.0)
    
    total_payments = v["total_payments"]
    total_payments=(float( total_payments) if  total_payments not in ["NaN"] else 0.0)
        
    total_stock_value = v["total_stock_value"]
    total_stock_value=(float( total_stock_value) if total_stock_value not in ["NaN"] else 0.0)        
                     
    v["incentives"] =   (float(  bonus) + float( long_term_incentive) + float(  exercised_stock_options))
                                 
    incentives= v["incentives"] 
            
    v["total_money"] =  (float( total_payments) + float( total_stock_value))
                         
    total_money=  v["total_money"]  
             
    v["bonus_ratio"] = round((float( bonus) / float(total_money)  if
                            bonus not in [0] and total_money not in [0] else 0.0),2)
                           
        
    v["incentives_ratio"] =  (round((float(incentives) / float( total_money) if
                            incentives not in [0] and total_money not in [0] else 0.0),2))
    

features_list.append("incentives") 
features_list.append("total_money")
features_list.append("bonus_ratio")
features_list.append("incentives_ratio")


# In[79]:


data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df  = data_df.loc[ :,( 'incentives', 'incentives_ratio',
                            'total_money','bonus_ratio', 
                            'salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees', 'total_payments',
                               'poi',     
                                'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi','email_address')]
data_df= data_df.replace('NaN', 0.0)
#data_df.head()


# We create some features and we analyse the correlation with poi features
# Other new features created with PCA
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

# In[80]:


from sklearn.decomposition import PCA
#features_list 'salary', 'bonus', 'long_term_incentive',
 #                             , ,
  #                             'other', 'expenses','director_fees',  'total_payments',
   #                            'exercised_stock_options', 'restricted_stock',
    #                           'restricted_stock_deferred', 'total_stock_value',
     #                          'from_messages', 'to_messages', 
      #                         'from_poi_to_this_person', 'from_this_person_to_poi',
       #                        'shared_receipt_with_poi','email_address', 'poi'
        
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


data_df  = data_df .loc[ :,( 'incentives', 'incentives_ratio','incentive_f',
                            'long_term_incentive','exercised_stock_options',
                              'bonus', 'bonus_ratio',
                            'salary','total_money','payment_f','payment_2','payment_tt','total_payments',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees', 
                               'poi',     
                                 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi','email_address','emails_exc')]


# We check the correlation between poi,'total_money', 'emails_exc' and all the features

# In[81]:


corr = data_df.corr() * 100
corr =corr.loc[ :,( 'poi', 'total_money', 'emails_exc')]
corr= round((corr.sort_values('poi',ascending=False)),0)
corr


# The correlation features with better correlation with poi are bonus,incentives, salary , payment_f , shared_receipt_with_poi , incentive_f, exercised_stock_options, total_stock_value, total_money , emails_exc,  
# total_payments, to_messages , from_poi_to_this_person 
# 
# The decision is to use: 'salary', 'bonus', 'incentives', 'expenses', 'other', 'total_payments', total_money ,'total_stock_value', 'shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi' and 'poi'   

# Now that we have reduce the features and create new ones lets see the dataset again

# In[82]:


features_list=['salary', 'bonus','incentives',   'expenses', 'other',
               'long_term_incentive',  'exercised_stock_options',
               'total_payments', 'total_stock_value', 'total_money',
               'shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi', 'poi']

data_df= data_df.loc[ :,(features_list )]  


data_df.head()


# In[83]:


round(data_df.describe(),0)


# Lets review the level of nan for the features now.
# We need to remove the new features to get the real level of Nan as the new features has not Nan but 0

# In[84]:


data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
   
for name in list_out:
    data_dict.pop(name,0)
    
#data_dict['LAY KENNETH L']

my_dataset = data_dict
data_df = pd.DataFrame.from_dict(data_dict, orient='index')

features_list=['salary', 'bonus', 'expenses', 'other',
               'long_term_incentive',  'exercised_stock_options',
               'total_payments', 'total_stock_value', 
               'shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi', 'poi']

data_df= data_df.loc[ :,(features_list )]  


# In[85]:


nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
round((nan_summary.sort_values('nan-proportion',ascending=False)),2)


# Lets see if still we have entries with high level of Nan.

# In[86]:


data_df1 = data_df.loc[ :,(features_list )]  

data_df1t=data_df1.T
        
nan_summary = pd.DataFrame({'size': data_df1t.count(),
                            'no-nan': data_df1t.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df1t.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
data_nan=(round((nan_summary.sort_values('nan-proportion',ascending=False)),2))
data_nan.head(25)


# We have some of them in the list of entries with director fees payment, as we do not use that feature, lets check which of them are in the list regarding top on Nan
# 
# CHAN RONNIE False 98784 -98784 NaN NaN NaN NaN NaN NaN . Ronnie Chan has been identified before as possible entry to be deleted, now that he has only 2 features (one is poi), we can confirm he does not added value to the analysis
# 
# SAVAGE FRANK False 125034 -121284 NaN 3750 NaN NaN NaN NaN . His payments comes from director fees, we do not use that information in our final data set then we delete him
# 
# WINOKUR JR. HERBERT S False 108579 -25000 NaN 84992 NaN NaN NaN NaN . His payments comes from director fees, we do not use that information in our final data set then we delete him
# 
# MENDELSOHN JOHN False 103750 -103750 NaN 148 NaN NaN NaN NaN . His payments comes from director fees, we do not use that information in our final data set then we delete him
# 
# MEYER JEROME J False 38346 -38346 NaN 2151 NaN NaN NaN NaN . His payments comes from director fees, we do not use that information in our final data set then we delete him
# 
# BLAKE JR. NORMAN P False 113784 -113784 NaN 1279 NaN NaN NaN NaN . His payments comes from director fees, we do not use that information in our final data set then we delete him
# 
# POWERS WILLIAM False 17500 -17500 NaN NaN NaN ken.powers@enron.com 0 0  . His payments comes from director fees, we do not use that information in our final data set then we delete him

# In[87]:


data_dict['CLINE KENNETH W']


# CLINE KENNETH W we found a mention to this person in the "Official Summary of Security Transactions and Holdingsok". His role is not clear. The information avalaible is on the features we decided to keep then we keep him in our dataset

# In[88]:


data_dict['WODRASKA JOHN']


# WODRASKA JOHN his payments comes from other, then we keep him in our data set as this feature is used.
# 
# Woody Wodraska is a veteran of the water industry and started his professional career with the South Florida Water Management District (SFWMD). He was then recruited to serve as the CEO of the Metropolitan Water District of Southern California (MWD). Wodraska has worked in a variety of positions in the private sector of the water industry.  This included Azurix, a subsidiary of ENRON and a stint with large to mid-sized Architectural & Engineering Consulting firms, specializing in water.  Presently he is President of Wodraska Partners, a small firm specializing in providing comprehensive problem solving strategies for public and private entities in the water industry https://www.verdexchange.org/profile/john-woody-wodraska

# In[89]:


data_dict['URQUHART JOHN A']


# URQUHART JOHN A payments comes from expenses, then we keep him in our data set as this feature is used.
# 
# He was Enron board memeber and got some payment for consulting services. "The report also says Enron's executives compromised the independence of some board members with consulting payments. Enron paid board member John Urquhart $493,914 for consulting in 2000. "
# https://www.theglobeandmail.com/report-on-business/us-senate-blasts-enron-directors/article25694090
# 

# we reload again our dataframe cleaning this other ientries 

# In[90]:


data_dict['PEREIRA PAULO V. FERRAZ']


# PEREIRA PAULO V. FERRAZ was Enron board memeber. He got payments on expenses as he was living abroad (Brasil) and got expenses when traveling for Board meetings, then we keep him in our data set as this feature is used.

# In[91]:


data_dict['WAKEHAM JOHN']


# WAKEHAM JOHN was Enron board memeber. He got payments on expenses as he was living abroad (UK) and got expenses when traveling for Board meetings, then we keep him in our data set as this feature is used.
# 
# "Three of the six departing directors live overseas: Ronnie C. Chan, 52, chairman of Hong Kong real estate firm Hang Lung Group; Paulo V. Ferraz Pereira, 47, executive vice president of Brazil’s Grupo Bozano banking firm; and Lord John Wakeham, 69, former secretary of state for energy in Britain."
# https://www.latimes.com/archives/la-xpm-2002-feb-13-fi--board13-story.html
# 

# In[92]:


data_dict['SCRIMSHAW MATTHEW']


# SCRIMSHAW MATTHEW was Enron . He got payments on expenses as he was living abroad (UK) and got expenses when traveling for Board meetings, then we keep him in our data set as this feature is used.
# 
# 
# I found in the net a mail exchange with a poi
# http://www.enron-mail.com/email/skilling-j/sent/Re_UK_CEOs_6.html
# but that does not appear in our data set.
# That means again that the data quality of our data set is not 100% accurate, what is the normal situation when working with real data and human data cranching.
# 
# and the reason why the entries with payment on director fees has amounts on stock is behind a company requirement
# https://www.latimes.com/archives/la-xpm-2002-feb-13-fi--board13-story.html
# "Non-employee Enron directors each have been paid $50,000 in service fees annually, plus $1,250 for each board meeting attended. The directors were required to defer 50% of their base annual service fee into an Enron stock plan."
# 

# Then after that anlysis we decide to remove from the data set the next entries
# CHAN RONNIE 
# SAVAGE FRANK 
# WINOKUR JR. HERBERT S 
# MENDELSOHN JOHN 
# MEYER JEROME J 
# BLAKE JR. NORMAN P 
# POWERS WILLIAM 

# Lets do that analysis again but removing all the entries identified until now 

# In[93]:


list_out=['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E','GRAMM WENDY L',
          'WHALEY DAVID A','WROBEL BRUCE', 'BELFER ROBERT', 'BHATNAGAR SANJAY', 
          'KAMINSKI WINCENTY J','SHAPIRO RICHARD S','KEAN STEVEN J','FASTOW ANDREW S',
          'KOPPER MICHAEL J','YEAGER F SCOTT','HIRKO JOSEPH',
          'CHAN RONNIE', 'SAVAGE FRANK', 'WINOKUR JR. HERBERT S', 'MENDELSOHN JOHN',
          'MEYER JEROME J', 'BLAKE JR. NORMAN P','POWERS WILLIAM'
         ]
features_list=['salary', 'bonus',  'expenses', 'other',
               'long_term_incentive',  'exercised_stock_options',
               'total_payments', 'total_stock_value', 
               'shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi', 'poi']

os.chdir(r'C:\Users\sanchez_sanc\Desktop\data_analyst\Curso\4_Machine_learning\Curso_PY_3\final_project') 

data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
   
for name in list_out:
    data_dict.pop(name,0)
    
data_df = pd.DataFrame.from_dict(data_dict, orient='index')   



data_df= data_df.loc[ :,(features_list )] 

#nan counting
nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
round((nan_summary.sort_values('nan-proportion',ascending=False)),2)    
    

#data_df = pd.DataFrame.from_dict(data_dict, orient='index')


# In[94]:


data_df1 = data_df.loc[ :,(features_list )]  

data_df1t=data_df1.T
        
nan_summary = pd.DataFrame({'size': data_df1t.count(),
                            'no-nan': data_df1t.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df1t.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
data_nan=(round((nan_summary.sort_values('nan-proportion',ascending=False)),2))
data_nan.head(25)


# The data set seams good now. Lets include the new features and continue with the analysis 

# In[95]:



for v in data_dict.values():
    
    bonus = v["bonus"]
    bonus=(float( bonus) if bonus not in ["NaN"] else 0.0)

    long_term_incentive = v["long_term_incentive"]
    long_term_incentive=(float( long_term_incentive) if long_term_incentive not in ["NaN"] else 0.0)
    
    exercised_stock_options =v["exercised_stock_options"] 
    exercised_stock_options=(float( exercised_stock_options) if exercised_stock_options not in ["NaN"] else 0.0)
    
    total_payments = v["total_payments"]
    total_payments=(float( total_payments) if  total_payments not in ["NaN"] else 0.0)
        
    total_stock_value = v["total_stock_value"]
    total_stock_value=(float( total_stock_value) if total_stock_value not in ["NaN"] else 0.0)        
                     
    v["incentives"] =   (float(  bonus) + float( long_term_incentive) + float(  exercised_stock_options))
                                 
    incentives= v["incentives"] 
            
    v["total_money"] =  (float( total_payments) + float( total_stock_value))
                         
    total_money=  v["total_money"]  
     
features_list.append("incentives") 
features_list.append("total_money")

# data frame
data_df = pd.DataFrame.from_dict(data_dict, orient='index')

data_df  = data_df .loc[ :,(features_list )]
data_df.info()


# Now we have a data set with 124 entries and 13 features plus 'poi' identification with mising data lower than 50%.
# Let see the main statistic information

# In[96]:


data_df= data_df.replace('NaN', 0.0)
round(data_df.describe(),0)


# And now a check of the correlations

# In[109]:


corr = data_df.corr() * 100
corr =corr.loc[ :,( 'poi', 'total_money', 'from_poi_to_this_person')]
corr= round((corr.sort_values('poi',ascending=False)),0)
corr


# Let see the relation between incentives and emails exchange with poi

# In[98]:


bicolor_plot("incentives", "from_poi_to_this_person", "Incentives", "Mail from Poi",20000000,200)


# Let see who are the outlier in "from_poi_to_this_person"

# In[110]:


features_mails=['shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi','incentives','poi']
mails_exc= data_df.loc[ :,(features_mails)] 
mails_exc =mails_exc .sort_values('from_poi_to_this_person',ascending=False)
mails_exc.head(3)


# Here some details about John Lavorato, considered non Poi.
# 
# The doubts were raised about Mr Skilling's testimony on the same day as CNN revealed that some 500 Enron staff had received windfalls ranging from $1,000 to $5m. The payments were made to retain staff as the firm faced collapse. To get the cash, the staff agreed to stay for 90 days.
# 
# The highest payment of $5m went to John Lavorato, who ran Enron's energy trading business, while Louise Kitchen, the division's British-born chief operating officer, pocketed $2m. Both have taken up new jobs with UBS Warburg, the investment bank that now owns the division.
# https://www.theguardian.com/business/2002/feb/11/corporatefraud.enron1
# https://archive.fortune.com/magazines/fortune/fortune_archive/2005/03/07/8253428/index.htm
# 
# Despite being an outlier in exchange mails with poi I consider he should remains in the data set as was one of the person that gto extra money to stay in the company during the collapse.

# In[100]:


bicolor_plot("incentives", "from_this_person_to_poi", "Incentives", "Mail to Poi",20000000,150)


# Let see who are the outlier in "from_this_person_to_poi"

# In[111]:


features_mails=['shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi','incentives','poi']
mails_exc= data_df.loc[ :,(features_mails)] 
mails_exc =mails_exc .sort_values('from_this_person_to_poi',ascending=False)
mails_exc.head(3)


# DELAINEY DAVID W  is considered as Poi then we keep in the data set and the second one is John Lavorato, see before.

# Let see if we have same outliers when we consider only bonus

# In[101]:


bicolor_plot("bonus", "from_poi_to_this_person", "Bonus", "Mail from Poi",4500000,200)


# In[102]:


bicolor_plot("bonus", "from_this_person_to_poi", "Bonus", "Mail to Poi",45000000,150)


# In[113]:


features_mails=['shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi','bonus','poi']
mails_exc= data_df.loc[ :,(features_mails)] 
mails_exc =mails_exc .sort_values('from_poi_to_this_person',ascending=False)
mails_exc.head(3)


# In[ ]:


features_mails=['shared_receipt_with_poi','from_poi_to_this_person', 'from_this_person_to_poi','bonus','poi']
mails_exc= data_df.loc[ :,(features_mails)] 
mails_exc =mails_exc .sort_values('from_this_person_to_poi',ascending=False)
mails_exc.head(3)


# We got the same people, and that has sense as one important component of incentive is bonus.
# 
# Let see them more in detail

# In[107]:


outl_mails=data_df.T
people_list=['DELAINEY DAVID W', 'LAVORATO JOHN J', 'BECK SALLY W', 'DIETRICH JANET R','KITCHEN LOUISE']

outl_mails =outl_mails.loc[ :,(people_list)] 
outl_mails


# BECK SALLY W  Ms. Beck was with Enron Corp for 10 years from 1992 to 2002. Her last position was Managing Director of Enron Corp and Chief Operating Officer of Enron Networks, an internal services company that included Information Technology and Trading Operations for financial and physical commodity trading. In that position, Ms. Beck was responsible for 2,100 employees worldwide
# https://relationshipscience.com/person/sally-beck-111369206
# sally.beck is within the top three highest ranking individuals in all centrality
# measures, except for eigenvector centrality. Her role in Enron was being a Chief
# Operating Officer (COO); responsible for the daily operation of the company
# and often reports directly to the Chief Executive Officer (CEO). Her result
# correspond to expectations of her role: a lot of sent and received messages to
# handle the daily operation
# https://www.iccs-meeting.org/archive/iccs2018/papers/108620361.pdf
# 
# DIETRICH JANET R: this is not poi but it seams the reason why was not cleared as we can read in the web
# At one point, Mr. Delainey appeared to lessen his insistence that the retail reorganization was a fraud. When pressed on whether Janet Dietrich, a top lieutenant of his, was aware of the alleged fraud, Mr. Delainey hesitated and appeared uncomfortable about possibly implicating someone who hasn't been criminally charged. Thundered Mr. Petrocelli: "If you're not sure Ms. Dietrich, a senior executive at the company, understood a fraud was being committed, how could anyone?"
# https://www.wsj.com/articles/SB114123416916986639
# Defense attorney Daniel Petrocelli said the government had an advantage in calling witnesses.
# “There were many members of enron’s senior management, many people, very important information that would have exonerated fully all of the positions that Mr. Skilling has been taking in this case. People like Lou Pai, people like Greg Whalley, people like Rich Buy, people like Janet Dietrich, and I could go on. They should have been here. A trial is supposed to be a place where the truth can be found. We wanted these people to come and testify. The government would not allow that to happen.”
# https://www.houstonpublicmedia.org/articles/news/2006/05/08/2700/monday-may-8th-2006/
# 
# KITCHEN LOUISE was a young British trader spearheading Enron’s entry into Europe’s energy markets. She wasn’t a top executive and hadn’t even turned 30. But Kitchen cooked up a plan for the company’s online trading operation.
# She got, as J. Lavorato some money to stay in the company
# The doubts were raised about Mr Skilling's testimony on the same day as CNN revealed that some 500 Enron staff had received windfalls ranging from $1,000 to $5m. The payments were made to retain staff as the firm faced collapse. To get the cash, the staff agreed to stay for 90 days.
# 
# The highest payment of $5m went to John Lavorato, who ran Enron's energy trading business, while Louise Kitchen, the division's British-born chief operating officer, pocketed $2m. Both have taken up new jobs with UBS Warburg, the investment bank that now owns the division.
# https://www.theguardian.com/business/2002/feb/11/corporatefraud.enron1

# Let see the outlier in the dataset again to be sure we identify all them 

# In[114]:


IQR = data_df.quantile(q=0.75) - data_df.quantile(q=0.25)
first_quartile = data_df.quantile(q=0.25)
third_quartile = data_df.quantile(q=0.75)
outliers = data_df[(data_df>(third_quartile + 1.5*IQR) ) | (data_df<(first_quartile - 1.5*IQR) )].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)
outliers.head()


# We have seen all them before and we have considered they should remain in the data set.

# We are ready now to move to the machine learning with our new features and clean data set
