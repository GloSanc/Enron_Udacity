#!/usr/bin/env python
# coding: utf-8

# # 1. DATA ADQUISITION  - 
# First thing to do is to create a dictionary with the features we need and we check the keys and the values of one key.

# In[31]:


import os 

os.chdir(r'C:\Users\sanchez_sanc\Desktop\data_analyst\Curso\4_Machine_learning\Curso_PY_3\tools') 

import sys
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#import warnings
#warnings.filterwarnings('ignore')

### Loading the dictionary containing the dataset
os.chdir(r'C:\Users\sanchez_sanc\Desktop\data_analyst\Curso\4_Machine_learning\Curso_PY_3\final_project') # the directory where you want to go

sys.path.append("../Curso_PY_3/final_project/")
data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

import pandas as pd
data_df = pd.DataFrame.from_dict(data_dict, orient='index')


# # 2. Data check: cross check figures, clean NaN, identify outliers and clean data set, visualize data

# Lets make a first old school check to ensure that total payment and total stock value are the sum of the different payments and stock features respectively

# In[9]:


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

# In[10]:


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


# There is only one register where the sum of stocks informations are not equal to the total stock value.
# And that is for the same person, Belfer Robert.
# Lets go to the pdf with the insider payment to check where the issue comes from.

# In[11]:


data_dict['BELFER ROBERT']


# In fact, none of the informations about payments or stock are right foer R. Belfer.
# But who is R. Belfer and what was his role in the colapse?
# Belfer was chairman of Belco Petroleum Corp. that was merged into one of the predecessors of Enron Corp, the Omaha, Nebraska-based InterNorth, Inc., and the Belfer family received a sizeable equity stake in the transaction, eventually becoming Enron's largest shareholder. Belfer served the board of directors of Enron and was estimated to have held over 16 million Enron shares as of August, 2000. However, he was reported to be reticent on the board and was not involved in the operations of the company. He resigned from the board in June, 2002.
# 
# We have two options, we delete his register from the data set or we correct them
# As no information related to mails is avalaible and the financial data are not relevant neither for payment or stock, the best option will be delete this register

# In[12]:


data_dict['BHATNAGAR SANJAY']


# In fact, none of the informations about payments are right, and we see that stock options neither.
# But who is S. Bhatnagar and what was his role in the colapse?
# Mr. Bhatnagar has served as the chairman and CEO of Enron South Asia where his responsibilities included developing and financing energy infrastructure. He was the lead developer of the 2,184 megawatt Dabhol Power project in India, which began commercial operations in May 1999.
# 
# We have two options, we delete his register from the data set or we correct them
# The information related to mails is that he send 29 mails and only 1 to a poi. 

# Lets analyse now the total payments and the total stock value to detect outliers.
# Firts lets remove the ones identified until now: "TOTAL", "THE TRAVEL AGENCY IN THE PARK",'BELFER ROBERT' and 'BHATNAGAR SANJAY'.

# In[13]:


clean_data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
clean_data_dict.pop("TOTAL", 0)
clean_data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
clean_data_dict.pop('BELFER ROBERT', 0)
clean_data_dict.pop('BHATNAGAR SANJAY', 0)

data_c_df = pd.DataFrame.from_dict(clean_data_dict, orient='index')
data_c_df= data_c_df.replace('NaN', 0.0)


# In[14]:


# 2D charts
import matplotlib.pyplot as plt

### the input features we want to use 

feature_1 = "total_stock_value"
feature_2 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(clean_data_dict, features_list ) # clan_data_dict se hace mas tarde aqui es data_dict
poi, finance_features = targetFeatureSplit( data )

for f1, f2 in finance_features:
    plt.scatter( f1, f2, )
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
plt.show()


# We have one outlier, lets see who is this rich man

# In[15]:


tt_paym = data_c_df.sort_values('total_payments',ascending=False)
tt_paym.head(2)


# In[16]:


tt_paym = data_c_df.sort_values('total_stock_value',ascending=False)
tt_paym.head(2)


# Lay is an outlier but not to be removed from our data set as he is the main Poi.
# Kenneth Lee Lay (April 15, 1942 – July 5, 2006) was the founder, CEO and Chairman of Enron and was heavily involved in the Enron scandal. Lay was indicted by a grand jury and was found guilty of 10 counts of securities fraud in the trial of Kenneth Lay and Jeffrey Skilling.Lay died in July 2006 while vacationing in his house near Aspen, Colorado, three months before his scheduled sentencing. A preliminary autopsy reported Lay died of a myocardial infarction (heart attack) caused by coronary artery disease; his death resulted in a vacated judgment.
# Lay's company, Enron, went bankrupt in 2001. At the time, this was the biggest bankruptcy in U.S. history. In total, 20,000 employees lost their jobs and in many cases their life savings. Investors also lost billions of dollars. On July 7, 2004, Lay was indicted by a grand jury in Houston, Texas, for his role in the company's failure. Lay was charged, in a 65-page indictment, with 11 counts of securities fraud, wire fraud, and making false and misleading statements. The Trial of Kenneth Lay and Jeffrey Skilling commenced on January 30, 2006, in Houston
# Lay insisted that Enron's collapse was due to a conspiracy waged by short sellers, rogue executives, and the news media.On May 25, 2006, Lay was found guilty on six counts of conspiracy and fraud by the jury. In a separate bench trial, Judge Lake ruled that Lay was guilty of four additional counts of fraud and making false statements. Sentencing was scheduled for September 11, 2006 and rescheduled for October 23, 2006.
# 
# We are going to analyse again the salary and the stock withou Lay to detect any outlier or mistake in the data

# In[17]:


# 2D charts
import matplotlib.pyplot as plt

# remove Lay Kenneth
clean_data_dict.pop("LAY KENNETH L", 0)

### the input features we want to use 

feature_1 = "total_stock_value"
feature_2 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(clean_data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
plt.show()


# There are people with salary 0 but payments and people with payments bigger than salary but not seams to be outlier in payment or stock value

# Lets see if salary is the main source of payment for the people in the data set

# In[18]:


### the input features we want to use 
feature_1 = "salary"
feature_2 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(clean_data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

for f1, f2 in finance_features:
    plt.scatter( f1, f2, )
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
plt.show()


# There are some payments on top of salary and for some there are not salary but payments but does not seam any outlier 

# Lets analyse now the mails exchange visualy to see if outliers including Lay

# In[19]:


clean_data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
clean_data_dict.pop("TOTAL", 0)
clean_data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
clean_data_dict.pop('BELFER ROBERT', 0)
clean_data_dict.pop('BHATNAGAR SANJAY', 0)

data_c_df = pd.DataFrame.from_dict(clean_data_dict, orient='index')
data_c_df= data_c_df.replace('NaN', 0.0)


# Lets check that the entries with no email adress does no have values in emails data

# In[20]:


# Define what is email data
email_data=['email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi'] 

# Create data frame of email with email data for the people that has email address (email address not 'NaN') 
# and replace NaN by 0.0  

emails = data_c_df.loc[:, 'email_address']  == 0
emails_c_df = data_c_df.loc[emails]

emails = emails_c_df.loc[ :,('poi','email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]

emails = emails.replace('NaN', 0.0)

#emails_df = emails_df.replace('NaN', 0.0)

by_from = emails.sort_values('from_messages',ascending=False)
by_from.head(35)


# We can see that all the entries with no email adress are non poi. Lets see if they have financial data

# In[21]:


# Define what is email data
email_data=['email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi'] 

# Create data frame of email with email data for the people that has email address (email address not 'NaN') 
# and replace NaN by 0.0  

emails = data_c_df.loc[:, 'email_address']  == 0
emails_c_df = data_c_df.loc[emails]

emails = emails_c_df.loc[ :,('poi','total_payments', 'total_stock_value','email_address', 'salary', 'deferral_payments',  'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income',  'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees')]

emails = emails.replace('NaN', 0.0)

#emails_df = emails_df.replace('NaN', 0.0)

by_from = emails.sort_values( 'total_payments',ascending=True)
by_from.head(35)


# There are several with financial data.
# We have to keep in mind later when we will select the features that we need to remove the entries with no values in the remaining and created features to avoid byas.
# Let see know the amounts of mails exchanged

# In[22]:


# Define what is email data
email_data=['email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi'] 

# Create data frame of email with email data for the people that has email address (email address not 'NaN') 
# and replace NaN by 0.0  

emails = data_c_df.loc[:, 'email_address']  != 'NaN'
emails_c_df = data_c_df.loc[emails]

emails = emails_c_df.loc[ :,('poi','email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]

emails = emails.replace('NaN', 0.0)

#emails_df = emails_df.replace('NaN', 0.0)

by_from = emails.sort_values('from_messages',ascending=False)
by_from.head()


# In[34]:


by_from = emails.sort_values('to_messages',ascending=False)
by_from.head(15)


# To messages and from messages have a max over 10 thousand message, I am curious to know who has send and got those level of messages

# In[35]:


data_dict['KAMINSKI WINCENTY J']


# Seams that W. Kaminsky is a very prolific emails sender, but not a Poi neither big payments neither big stock values. Lets go to the web to discover who is this person.
# 
# Vincent Julian Kaminski was born in Poland and worked as the Managing Director for Research at the failed energy trading corporation Enron until 2002. In this capacity he led a team of approximately fifty analysts who developed quantitative models to support energy trading. In the months preceding Enron’s bankruptcy Kaminski repeatedly raised strong objections to the financial practices of Enron’s Chief Financial Officer, Andrew Fastow, designed to fraudulently conceal the company’s burgeoning debt.
# 
# And we found in the net the origin of this amount of data: "Though much private data has been removed, browsing hundreds of e-mails in Kaminski’s “sent” folder, I found a home phone number, his wife’s name, and an unflattering opinion he held of a former colleague. I also got the sense that he had been long, long overdue for the promotion he received in 2000. At the time the e-mails were first released, Kaminski, the manager of about 50 employees at Enron, said he was most disturbed to see his back-and-forth communications about HR complaints and job candidate evaluations become public. A job candidate he once interviewed got upset after their release.".
# 
# @ https://www.technologyreview.com/2013/07/02/177506/the-immortal-life-of-the-enron-e-mails/
# 
# Regarding the amount of mails, we need to keep in mind that the emails has been cleaning but not depurate, that means there are mails not relevant for the investigation in the data set. 
# 
# Now the mistery reveal, I decide to remove this person from the data as could be misleading for the data analysis.
# 
# Lets investigate now Richard Shapiro and Steven Kean

# In[36]:


data_dict['SHAPIRO RICHARD S']


# Richard Shapiro: Vice President of Regulatory Affairs. He is not consifered as Poi then I prefer to remove him from the data set.
# 
# Lets see Steven J. Kean

# In[26]:


data_dict['KEAN STEVEN J']


# He is not poi, but aparently exchange a lot of emails with Poi 

# In[37]:


by_from = emails.sort_values('from_this_person_to_poi',ascending=False)
by_from.head()


# In fact, he si the 3rd that send the most mail to poi, lets investigate in the net.
# Steven J. Kean, was Enron’s former senior Vice President of Government Affairs. 
# There is a very interesting analysis about the mails of Mr. Kean at Enron data set @ 
# https://www.researchgate.net/publication/327252947_Security_Threats_for_Big_Data_An_Empirical_Study
#    
# He is not considered as Poi then I prefer to remove him from the data set despite he is on the top people who send mails to Poi.

# Then we have now some outliers to be traeted
# Lets remove form the data set 'THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 'BELFER ROBERT', 'BHATNAGAR SANJAY', 'KAMINSKI WINCENTY J', 'SHAPIRO RICHARD S' and 'KEAN STEVEN J'

# In[38]:


clean_data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
clean_data_dict.pop("TOTAL", 0)
clean_data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
clean_data_dict.pop('BELFER ROBERT', 0)
clean_data_dict.pop('BHATNAGAR SANJAY', 0)
clean_data_dict.pop('KAMINSKI WINCENTY J', 0)
clean_data_dict.pop('SHAPIRO RICHARD S',0)
clean_data_dict.pop('KEAN STEVEN J', 0)
clean_data_dict.keys()


# In[49]:


import pandas as pd
data_df = pd.DataFrame.from_dict(clean_data_dict, orient='index')
data_df.head()

Lets create again a data frame, clean the "NaN", fix number format to float, organize the columns and see if any other information seams strange
# Lets see now the information regarding the different features

# In[50]:


nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
round((nan_summary.sort_values('nan-proportion',ascending=False)),2)


# We see that we have some categories that provides low level of data, we should investigate if those should remains as features or not.
# As seen during data adquisition, loan advances could be removed or integrate with another category in a new feature.
# Perhaps we could create a compound feature with some of the payment / stock categories but that is part or another discussion, lets continue now with data checking and visualization.

# Let see some statistic from the data to try to detect further anomalies

# In[51]:


data_m_df= data_df.replace('NaN', 0.0)
round(data_m_df.describe(),0)


# The total payment average is 2.2m$ with a Q3 on 1.9m$ (lower than the average) with a maximum value of 103.56m$. Total stock value has an average of 2.96m$ and a Q3 at 2.30 $ (lower than the average) with a maximum value of 49.1m$. That is thanks to our outlier K. Lay.
# 

# Let see some details of the person of interes (poi=True)

# In[56]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poi =poi_df.loc[:, ('poi','salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments', 'restricted_stock_deferred','exercised_stock_options', 
                 'restricted_stock', 'total_stock_value', 'email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]


# In[55]:


pnan_summary = pd.DataFrame({'size': poi.count(),
                            'no-nan': poi.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            poi.columns)
pnan_summary['nan-proportion'] =1-( pnan_summary['no-nan'] / pnan_summary['size'])
round((pnan_summary.sort_values('nan-proportion',ascending=False)),2)


# We have 18 person of interes with their mails, but 4 has NaN in from poi and to poi messages.
# Regarding financial features, Nan is not a real missing information but that for that category there was no payment or stock value

# Lets see what is behind poi data regarding total payment, stock and emails exchange

# In[72]:


poim =poi_df.loc[:, ('poi','total_payments','total_stock_value',
                    'from_messages', 'to_messages',
                    'from_poi_to_this_person', 'from_this_person_to_poi' )]


# In[76]:


poim_0= poim.replace('NaN', 0.0)
round(poim_0.describe(),0)


# We can see some outlier here. We know the outlier for total payments and stock but who are the outlier for emaisl exchange with poi?. 

# In[77]:


poim_0 = poim_0.sort_values('from_this_person_to_poi' ,ascending=False)
poim_0.head(len(poi))


# Lets see who are the outlier that send more than 600 mails to poi 
# DELAINEY DAVID W : CEO of Enron North America and Enron Enery Services. A critical prosecution witness against former Enron Corp. CEO Jeff Skilling may have admitted only to selling stock illegally three years ago, but a judge said Monday his wrongdoing at the fallen company went much deeper. "Any employee who had some sense of what was going on could be charged with that," U.S. District Judge Kenneth Hoyt said before he sentenced David Delainey to 2 1/2 years in prison for the insider trading count central to his October 2003 plea deal.
# 
# 
# And who are the ones without mails
# 
# FASTOW ANDREW S: Andrew Fastow was the chief financial officer of Enron Corporation, an energy trading company based in Houston, Texas, until he was fired shortly before the company declared bankruptcy. Fastow was one of the key figures behind the complex web of off-balance-sheet special purpose entities (limited partnerships which Enron controlled) used to conceal Enron’s massive losses in their quarterly balance sheets. By unlawfully maintaining personal stakes in these ostensibly independent ghost-entities, he was able to defraud Enron out of tens of millions of dollars. He was convicted and served a six-year prison sentence for charges related to those acts. These days he spends most of his time speaking about ethics and anti-fraud procedures at universities and events like the Annual Global Anti-Fraud Conference. https://www.gregorybufithis.com/2019/04/04/text-analytics-the-enron-data-set-and-andrew-fastow-get-a-new-role/
# Enron finance chief who received the CFO Excellence Award in the category of “capital structure management.”  In a  recent paper we can read "In his right hand, he was holding his CFO Excellence trophy. In his left hand was his prison identification card. He then raised both arms and said, “How is it possible to go from a CFO of the year to federal prison for doing the same deals?”" (https://www.cfo.com/risk-compliance/2019/04/first-person-andy-fastow-and-me/). This article talk about the importance of the Faston mails in different analysis then we do not undertand why not email from Andrew Fastow in the dataset but we have found another paper in the net that explain that his email networks consisting of 188 emails with 215 unique actors. This data comprises four differ- ent sources within the Enron data set as many of his contacts were external to the company and therefore not included in the email corpus. Perhaps that explain why no mails in this dataset
# (https://www.researchgate.net/figure/Andrew-Fastows-Enron-email-network_fig3_220295304)
# 
# KOPPER MICHAEL J  Michael Kopper, a former Enron Corp. executive convicted of helping ex-Chief Financial Officer Andrew Fastow skim millions of dollars from the energy trader. Kopper, 43, was the first Enron executive to cut a deal, pleading guilty to two conspiracy counts and cooperating with prosecutors pursuing other company officials. In 2006, a federal judge who heard Kopper testify sentenced him to 37 months, compared with the 15-year maximum term he agreed to in his plea.https://www.latimes.com/archives/la-xpm-2009-jan-03-fi-kopper3-story.html
# 
# YEAGER F SCOTT was a strategic business executive in Enron's Internet broadband section, has spent seven years trying to clear his name, one of the lesser-known in the Enron scandal. https://www.chron.com/business/enron/article/Cleared-Enron-defendant-warns-of-prosecutors-1741638.php
# 
# HIRKO JOSEPH Hirko, of Portland, Ore., joined Enron in 1997 when the company acquired Portland General Electric. He was chief financial officer of PGE, which had a fledgling telecommunications division. He became CEO of what grew into Enron Broadband Services, and former top Enron trader Kenneth Rice joined him as co-CEO for a year until Hirko left Enron in mid-2000. https://www.chron.com/business/enron/article/Former-Enron-broadband-chief-gets-prison-term-1622724.php
# 
# We can see that they play a role in the story but if we do not have information about mails, I am tempted to consider them as noise. 
# 
# Let see now some details related to financial features

# In[65]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poi =poi_df.loc[:, ('salary', 'bonus', 'other', 'long_term_incentive', 
                    'expenses', 'deferred_income','total_payments', 'total_stock_value',
                 'exercised_stock_options',  'restricted_stock' )]

poi = poi.sort_values('total_stock_value',ascending=False)

poi.head(len(poi))


# In[45]:


poi_0= poi.replace('NaN', 0.0)
round(poi_0.describe(),0)


# HIRKO JOSEPH does not have salary, the payments he got are mainly expenses, regardign stock he is the second to have more stock value.
# That does not justify to keep him in the data set
# Despite we have only 18 poi I would prefer to remove the ones which information missing will generate misliding interpretation
# 

# In[90]:


clean2_data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

clean2_data_dict.pop('TOTAL', 0)
clean2_data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
clean2_data_dict.pop('BELFER ROBERT', 0)
clean2_data_dict.pop('BHATNAGAR SANJAY', 0)
clean2_data_dict.pop('KAMINSKI WINCENTY J', 0)
clean2_data_dict.pop('SHAPIRO RICHARD S',0)
clean2_data_dict.pop('KEAN STEVEN J', 0)
clean2_data_dict.pop('FASTOW ANDREW S', 0) 
clean2_data_dict.pop('KOPPER MICHAEL J', 0)
clean2_data_dict.pop('YEAGER F SCOTT', 0)
clean2_data_dict.pop('HIRKO JOSEPH', 0)

clean2_data_dict.keys()

import pandas as pd
data_df = pd.DataFrame.from_dict(clean2_data_dict, orient='index')


data_df= data_df.loc[ :,('salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi','email_address', 'poi'
                          )]  

#data_df= data_df.replace('NaN', 0.0)
#round(data_df.describe(),0)

nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
round((nan_summary.sort_values('nan-proportion',ascending=False)),2)


# In[93]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poi =poi_df.loc[:, ('poi','salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments', 'restricted_stock_deferred','exercised_stock_options', 
                 'restricted_stock', 'total_stock_value', 'email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]

pnan_summary = pd.DataFrame({'size': poi.count(),
                            'no-nan': poi.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            poi.columns)
pnan_summary['nan-proportion'] =1-( pnan_summary['no-nan'] / pnan_summary['size'])
round((pnan_summary.sort_values('nan-proportion',ascending=False)),2)


# Now we have only 14 poi but the quality of the data has improve, we have 100% of information related to poi exchange mails.
# 
# See now the information related to the cleaned data

# In[2]:


data_df= data_df.replace('NaN', 0.0)
round(data_df.describe(),0)


# The level on Nan is high for some features. 
# I remove all the features with more than 70% of Nan in the person that are Poi and at total data set :  'director_fees' (0%, 10%), 'restricted_stock_deferred' (0%, 89%), 'loan_advances' (93%, 98%), ‘deferral_payments' (71%, 73%) 
# I remove 'deferred_income' because that is a negative value that represent a discount in the payment and it is missing in 65% of the records
# Regarding the emails, this project use the dataset corresponding to the counting of mails and not gone to the details of the text in the emails. Taking into account that the total amount of emails for each person has been cleaned in a more or less proper way, I decided not to use that total amount but only the mails with relation with poi, that means send to or received from poi. On top, there are 40% of persons without mails
