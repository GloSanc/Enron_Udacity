#!/usr/bin/env python
# coding: utf-8

# # 1. DATA ADQUISITION 
# First thing to do is to create a dictionary with the features we need and we check the keys and the values of one key.
# Lets create a data frame, clean the "NaN", fix number format to float, organize the columns and see if any other information seams strange, after removing 'THE TRAVEL AGENCY IN THE PARK', 'TOTAL' and correct 'BELFER ROBERT', 'BHATNAGAR SANJAY

# In[2]:


### Import some packages and modules to be used later
import pickle
import pandas as pd
import numpy as np

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from feature_format import featureFormat, targetFeatureSplit

from PML_1_data_adq import data_dict_cor 

### Loading the dictionary containing the dataset

data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

### running the function to remove the non person entries and correct the wrong values
### 'THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 'BELFER ROBERT', 'BHATNAGAR SANJAY'

data_dict_cor (data_dict)

my_dataset = data_dict

# Creating the data frame from the dictionary with the existing features

data_df = pd.DataFrame.from_dict(data_dict, orient='index')


# # 2. Data check: cleaning NaN, visualize data and check outliers

# ## 2.1. Data set after cleanning 

# In[3]:


nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
nan_summary=round((nan_summary.sort_values('nan-proportion',ascending=False)),2)


# In[4]:


nan_summary


# We have 144 entries and we see that we have some categories that provides low level of data, we should investigate if those should remains as features or not.
# Perhaps we could create a compound feature with some of the payment / stock categories but that is part or another discussion, lets continue now with some statistic.
# Let see now the level of nan for non poi, as all the changes has been done over non poi population

# In[5]:


non_poi = data_df.loc[:, 'poi'] == False
non_poi_df = data_df.loc[non_poi]

non_poi =non_poi_df.loc[:, ('poi','salary', 'deferral_payments', 'loan_advances', 'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments', 'restricted_stock_deferred','exercised_stock_options', 
                 'restricted_stock', 'total_stock_value', 'email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]

pnan_summary = pd.DataFrame({'size': non_poi.count(),
                            'no-nan': non_poi.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            non_poi.columns)
pnan_summary['nan-proportion'] =1-( pnan_summary['no-nan'] / pnan_summary['size'])
round((pnan_summary.sort_values('nan-proportion',ascending=False)),2)


# ## 2.2. Statistic for poi and non-poi
# Let see some statistic from the data to try to detect further anomalies

# In[6]:


data_m_df= data_df.replace('NaN', 0.0)
round(data_m_df.describe(),0)


# The total payment average is 2.2m with a Q3 on 1.9m (lower than the average) with a maximum value of 103.56m. Total stock value has an average of 2.96m and a Q3 at 2.3m (lower than the average) with a maximum value of 49.1m.

# Let see some details of the person of interes (poi=True) regarding total payment, stock and emails exchange

# In[7]:


# Poi data with Nan as 0
poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poim =poi_df.loc[:, ('poi','total_payments','total_stock_value',
                    'from_messages', 'to_messages',
                    'from_poi_to_this_person', 'from_this_person_to_poi' )]
poim_0= poim.replace('NaN', 0.0)
round(poim_0.describe(),0)


# The total payment average is 7.9m (5m bigger than dataset average) with a Q3 on 2.7m (lower than the average) with a maximum value of 103.56m. Total stock value has an average of 9.2m (7m bigger than dataset average) and a Q3 at 10.3m (bigger than the average) with a maximum value of 49.1m.  
# 
# Let see who are the those 18 Poi (Person of interest), their total payment, stock and emails exchange

# In[8]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poi =poi_df.loc[:, ('poi','total_payments','total_stock_value',
                     'email_address', 'from_messages', 'to_messages',
                    'from_poi_to_this_person', 'from_this_person_to_poi' )]
poi= poi.sort_values('total_payments',ascending=False)
poi.head(len(poi))


# In[9]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poi =poi_df.loc[:, ('poi','total_payments','total_stock_value',
                     'email_address', 'from_messages', 'to_messages',
                    'from_poi_to_this_person', 'from_this_person_to_poi' )]
poi= poi.sort_values('total_stock_value',ascending=False)
poi.head(1)


# The outlier in total_payments and total_stock_valueboth is LAY KENNETH L the CeO of the company.
# Regarding the mails we could see that we have big outliers in all categories
# We have 18 person of interes with their mails, but 4 has NaN in from poi and to poi messagess 
# Let see some details of payment and stock for poi persons

# In[10]:


poi = data_df.loc[:, 'poi'] == True
poi_df = data_df.loc[poi]

poi =poi_df.loc[:, ('salary', 'bonus', 'other', 'long_term_incentive', 
                    'expenses', 'deferred_income',
                    'exercised_stock_options',  'restricted_stock',
                    'total_stock_value','total_payments'
                 )]
poi= poi.sort_values('total_payments',ascending=False)
poi.head(len(poi))


# We can see that not all of them got long_term_incentive or exercised_stock_options.
# And there is one person without restricted_stock
# Lets see the statistic for all the features

# In[11]:


data_df1= data_df.replace('NaN', 0.0)
poi_0= poi.replace('NaN', 0.0)
round(poi_0.describe(),0)


# Let see now some details of non person of interes (poi=False) regarding total payment, stock and emails exchange

# In[12]:


npoi = data_df.loc[:, 'poi'] == False
npoi_df = data_df.loc[npoi]

npoim =npoi_df.loc[:, ('poi','total_payments','total_stock_value',
                    'from_messages', 'to_messages',
                    'from_poi_to_this_person', 'from_this_person_to_poi' )]
npoim_0= npoim.replace('NaN', 0.0)
round(npoim_0.describe(),0)


# The total payment average is 1.3m (lower than dataset average) with a Q3 on 1.7m (bigger than the average) with a maximum value of 17.3m. Total stock value has an average of 2m (lower than dataset average) and a Q3 at 2m (close to the average) with a maximum value of 23.8m. 
# Regarding the mails we could see that we have big outliers in all categories. The otulier for all the emails features belong to non-poi except for from_this_person_to_poi. Again we are confronted to the issue of emails cleanning.

# We are going to dig now in the financial features 

# ## 2.3.	Financial features: Data visualization, Outlier and low number of entries 
# 
# Lets analyse now in details the financial features.

# In[13]:


data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df= data_df.replace('NaN', 0.0)


# In[14]:


# 2D charts
import matplotlib.pyplot as plt

### the input features we want to use 

feature_1 = "total_stock_value"
feature_2 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list ) # clan_data_dict se hace mas tarde aqui es data_dict
poi, finance_features = targetFeatureSplit( data )

for f1, f2 in finance_features:
    plt.scatter( f1, f2, )
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
plt.show()


# We have one outlier, lets see who is this well paid man

# In[15]:


tt_paym = data_df.sort_values('total_payments',ascending=False)
tt_paym.head(2)


# In[16]:


tt_paym = data_df.sort_values('total_stock_value',ascending=False)
tt_paym.head(2)


# LAY KENNETH L is an outlier but not to be removed from our data set as he is the main Poi.
# Kenneth Lee Lay (April 15, 1942 – July 5, 2006) was the founder, CEO and Chairman of Enron and was heavily involved in the Enron scandal. Lay was indicted by a grand jury and was found guilty of 10 counts of securities fraud in the trial of Kenneth Lay and Jeffrey Skilling.Lay died in July 2006 while vacationing in his house near Aspen, Colorado, three months before his scheduled sentencing. A preliminary autopsy reported Lay died of a myocardial infarction (heart attack) caused by coronary artery disease; his death resulted in a vacated judgment.
# Lay's company, Enron, went bankrupt in 2001. At the time, this was the biggest bankruptcy in U.S. history. In total, 20,000 employees lost their jobs and in many cases their life savings. Investors also lost billions of dollars. On July 7, 2004, Lay was indicted by a grand jury in Houston, Texas, for his role in the company's failure. Lay was charged, in a 65-page indictment, with 11 counts of securities fraud, wire fraud, and making false and misleading statements. The Trial of Kenneth Lay and Jeffrey Skilling commenced on January 30, 2006, in Houston
# Lay insisted that Enron's collapse was due to a conspiracy waged by short sellers, rogue executives, and the news media.On May 25, 2006, Lay was found guilty on six counts of conspiracy and fraud by the jury. In a separate bench trial, Judge Lake ruled that Lay was guilty of four additional counts of fraud and making false statements. Sentencing was scheduled for September 11, 2006 and rescheduled for October 23, 2006.
# 
# We are going to analyse again the salary and the stock withou Lay to detect any outlier or mistake in the data

# In[17]:


# 2D charts
import matplotlib.pyplot as plt

# remove Lay Kenneth
data_dict.pop("LAY KENNETH L", 0)

### the input features we want to use 

feature_1 = "total_stock_value"
feature_2 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
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
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

for f1, f2 in finance_features:
    plt.scatter( f1, f2, )
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
plt.show()


# There are some payments on top of salary and for some there are not salary but payments but does not seam any outlier.
# 

# Lets see the details of the top low entries features: Loan advances and director fees.

# In[19]:


nan_summary=round((nan_summary.sort_values('nan-proportion',ascending=False)),2)
nan_summary


# Loan advances has only 3 entries with value (2% of the entries). 
# 
# If this feature seams not relevant, we could delete from our investigation. 

# In[20]:


data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
data_dict_cor (data_dict)
data_df = pd.DataFrame.from_dict(data_dict, orient='index')

loan = data_df.loc[:, 'loan_advances'] != 'NaN'
loan_df = data_df.loc[loan]
loan =loan_df.loc[:, ('poi','loan_advances', 'salary', 'deferral_payments',  'bonus',
                 'deferred_income',  'expenses', 'other', 'long_term_incentive',
                 'director_fees',  'total_payments')]
loan= loan.sort_values( 'loan_advances',ascending=False)
loan.head(3)


# We extract the information related to the entries with loan advances and we found that:
#  
# Kenneth Lay: he got a loan advances of 81m, is nearly 100% of all the loan advances. This advance makes his salary very low. This is misleading information. We should keep that in mind later to create a feature that combine several payments as each payment by itself could be misleading. That will contribute to reduce the amount features the algorithm should analyse and avoid misleading features. We will back later on this
# 
# There are two other people who got loan advances that are non Poi that means loan advances is not discriminant of Poi. That could indicate that is not relevant feature by itself.
# 
# Lets pass to investigate emails 

# ## 2.4. Email features: Data visualization and Outlier
# 
# Lets analyse now in details the emails features.

# In[21]:


data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
data_dict_cor (data_dict)
data_df = pd.DataFrame.from_dict(data_dict, orient='index')

data_df= data_df.replace('NaN', 0.0)


# In[22]:


### the input features we want to use 
feature_1 = "from_messages"
feature_2 = "to_messages"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

for f1, f2 in finance_features:
    plt.scatter( f1, f2, )
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
plt.show()


# Lets investigate who are the outlier in from and to messages

# In[23]:


# Define what is email data
email_data=['email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi'] 

# Create data frame of email with email data for the people that has email address (email address not 'NaN') 
# and replace NaN by 0.0  

emails = data_df.loc[:, 'email_address']  != 'NaN'
emails_df = data_df.loc[emails]

emails = emails_df.loc[ :,('poi','email_address', 'from_messages', 'to_messages', 'shared_receipt_with_poi',
            'from_poi_to_this_person', 'from_this_person_to_poi')]

emails = emails.replace('NaN', 0.0)

#emails_df = emails_df.replace('NaN', 0.0)

by_from = emails.sort_values('from_messages',ascending=False)
by_from.head()


# To messages and from messages have a max over 10 thousand message, I am curious to know some details over those outlier on emails

# KAMINSKI WINCENTY J Seams that W. Kaminsky is a very prolific emails sender, but not a Poi neither big payments neither big stock values. Lets go to the web to discover who is this person.
# 
# Vincent Julian Kaminski was born in Poland and worked as the Managing Director for Research at the failed energy trading corporation Enron until 2002. In this capacity he led a team of approximately fifty analysts who developed quantitative models to support energy trading. In the months preceding Enron’s bankruptcy Kaminski repeatedly raised strong objections to the financial practices of Enron’s Chief Financial Officer, Andrew Fastow, designed to fraudulently conceal the company’s burgeoning debt.
# 
# And we found in the net the origin of this amount of data: "Though much private data has been removed, browsing hundreds of e-mails in Kaminski’s “sent” folder, I found a home phone number, his wife’s name, and an unflattering opinion he held of a former colleague. I also got the sense that he had been long, long overdue for the promotion he received in 2000. At the time the e-mails were first released, Kaminski, the manager of about 50 employees at Enron, said he was most disturbed to see his back-and-forth communications about HR complaints and job candidate evaluations become public. A job candidate he once interviewed got upset after their release.".
# 
# @ https://www.technologyreview.com/2013/07/02/177506/the-immortal-life-of-the-enron-e-mails/
# 
# Regarding the amount of mails, we need to keep in mind that the emails has been cleaning but not depurate, that means there are mails not relevant for the investigation in the data set. 
# 
# Now the mistery reveal, we could think about to remove this person from the data as could be misleading for the data analysis.
# 
# Lets investigate now Richard Shapiro and Steven Kean

# In[24]:


by_from = emails.sort_values('to_messages',ascending=False)
by_from.head()


# SHAPIRO RICHARD S: Vice President of Regulatory Affairs. He is not consifered as Poi then I prefer to remove him from the data set.
# 
# Lets see Steven J. Kean: He is not poi, but aparently exchange a lot of emails with Poi 

# Let see who are the champions on sending mails to poi

# In[25]:


by_from = emails.sort_values('from_this_person_to_poi',ascending=False)
by_from.head(6)


# And the values in the different features for those champions

# In[28]:



people_list=['DELAINEY DAVID W', 'LAVORATO JOHN J', 'BECK SALLY W', 'DIETRICH JANET R','KITCHEN LOUISE']
by_from_t=data_df.T
by_from_t =by_from_t.loc[ :,(people_list)] 
by_from_t


# The first one is DELAINEY DAVID W that is a poi, but the next ones are non poi
# 
# LAVORATO JOHN J is considered non Poi but we found in the web that
# The doubts were raised about Mr Skilling's testimony on the same day as CNN revealed that some 500 Enron staff had received windfalls ranging from 1,000 to 5m. The payments were made to retain staff as the firm faced collapse. To get the cash, the staff agreed to stay for 90 days.
# 
# The highest payment of 5m went to John Lavorato, who ran Enron's energy trading business, while Louise Kitchen, the division's British-born chief operating officer, pocketed 2m. Both have taken up new jobs with UBS Warburg, the investment bank that now owns the division.
# https://www.theguardian.com/business/2002/feb/11/corporatefraud.enron1
# https://archive.fortune.com/magazines/fortune/fortune_archive/2005/03/07/8253428/index.htm
# 
# Despite being an outlier in exchange mails with poi I consider he should remains in the data set as was one of the person that gto extra money to stay in the company during the collapse.

# The 3rd one is KEAN STEVEN J, the Enron’s former senior Vice President of Government Affairs. 
# There is a very interesting analysis about the mails of Mr. Kean at Enron data set @ 
# https://www.researchgate.net/publication/327252947_Security_Threats_for_Big_Data_An_Empirical_Study
#    
# Knowing his postition in the company it is a bit surprising he is not considered as Poi, or perhaps for his position he provide information for the investigation in exchange of inmmunity. Who knows?

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
# KITCHEN LOUISE was a young British trader spearheading Enron’s entry into Europe’s energy markets. She wasn’t a top executive and hadn’t even turned 30. But Kitchen cooked up a plan for the company’s online trading operation.
# She got, as J. Lavorato some money to stay in the company
# The doubts were raised about Mr Skilling's testimony on the same day as CNN revealed that some 500 Enron staff had received windfalls ranging from $1,000 to $5m. The payments were made to retain staff as the firm faced collapse. To get the cash, the staff agreed to stay for 90 days.
# The highest payment of $5m went to John Lavorato, who ran Enron's energy trading business, while Louise Kitchen, the division's British-born chief operating officer, pocketed $2m. Both have taken up new jobs with UBS Warburg, the investment bank that now owns the division.
# https://www.theguardian.com/business/2002/feb/11/corporatefraud.enron1
# 
# DIETRICH JANET R: this is not poi but it seams the reason why was not cleared as we can read in the web
# At one point, Mr. Delainey appeared to lessen his insistence that the retail reorganization was a fraud. When pressed on whether Janet Dietrich, a top lieutenant of his, was aware of the alleged fraud, Mr. Delainey hesitated and appeared uncomfortable about possibly implicating someone who hasn't been criminally charged. Thundered Mr. Petrocelli: "If you're not sure Ms. Dietrich, a senior executive at the company, understood a fraud was being committed, how could anyone?"
# https://www.wsj.com/articles/SB114123416916986639
# Defense attorney Daniel Petrocelli said the government had an advantage in calling witnesses.
# “There were many members of enron’s senior management, many people, very important information that would have exonerated fully all of the positions that Mr. Skilling has been taking in this case. People like Lou Pai, people like Greg Whalley, people like Rich Buy, people like Janet Dietrich, and I could go on. They should have been here. A trial is supposed to be a place where the truth can be found. We wanted these people to come and testify. The government would not allow that to happen.”
# https://www.houstonpublicmedia.org/articles/news/2006/05/08/2700/monday-may-8th-2006/

# Let see know the information on mails on pois

# In[61]:


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
# HIRKO JOSEPH does not have salary, the payments he got are mainly expenses, regardign stock he is the second to have more stock value.
# That could justify to keep him in the data set even if he has not mails
# On top, we have only 18 poi and it is preferable do not delete any of them
# 

# We are going to visualize from and to message removing the outliers identified

# In[46]:


# 2D charts
import matplotlib.pyplot as plt

# remove outlier
data_dict.pop("KAMINSKI WINCENTY J", 0)
data_dict.pop("SHAPIRO RICHARD S", 0)
data_dict.pop("KEAN STEVEN J", 0)

### the input features we want to use 

feature_1 = "from_messages"
feature_2 = "to_messages"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

for f1, f2 in finance_features:
    plt.scatter( f1, f2, )
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
plt.show()


# ## 2.5. Dataset overview with and without specific entries

# Lets see how it looks like a data set without some of the entries we have identified as possible to be deleted

# In[ ]:


clean_data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

data_dict_cor (clean_data_dict)

list_out=['TOTAL','THE TRAVEL AGENCY IN THE PARK',
          'LOCKHART EUGENE E',
          'GRAMM WENDY L','WHALEY DAVID A','WROBEL BRUCE', 
          'KAMINSKI WINCENTY J','SHAPIRO RICHARD S','KEAN STEVEN J','FASTOW ANDREW S',
          'KOPPER MICHAEL J','YEAGER F SCOTT','HIRKO JOSEPH',
          'CHAN RONNIE', 'SAVAGE FRANK', 'WINOKUR JR. HERBERT S', 'MENDELSOHN JOHN',
          'MEYER JEROME J', 'BLAKE JR. NORMAN P','POWERS WILLIAM'
         ]

for name in list_out:
    clean_data_dict.pop(name,0)
    
import pandas as pd
data_df = pd.DataFrame.from_dict(clean_data_dict, orient='index')


data_df= data_df.loc[ :,('salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi','email_address', 'poi'
                          )]  

nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
round((nan_summary.sort_values('nan-proportion',ascending=False)),2)


# In[ ]:


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

# In[ ]:


data_df= data_df.replace('NaN', 0.0)
round(data_df.describe(),0)


# After all this analysis I decide to remove from the dataset only 2 entries

# In[ ]:


clean_f_data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))

clean_f_data_dict.pop('TOTAL', 0)
clean_f_data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


clean_f_data_dict.keys()

data_df = pd.DataFrame.from_dict(clean_f_data_dict, orient='index')
data_df= data_df.loc[ :,('salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi','email_address', 'poi'
                          )]  

nan_summary = pd.DataFrame({'size': data_df.count(),
                            'no-nan': data_df.applymap(lambda x: pd.np.nan if x=='NaN' else x).count()},
                            data_df.columns)
nan_summary['nan-proportion'] =1-( nan_summary['no-nan'] / nan_summary['size'])
round((nan_summary.sort_values('nan-proportion',ascending=False)),2)


# In[49]:


data_df= data_df.replace('NaN', 0.0)
round(data_df.describe(),0)


# Lets move now to the next section related to features analysis, creation and selection
