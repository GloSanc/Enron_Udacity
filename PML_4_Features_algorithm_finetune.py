#!/usr/bin/env python
# coding: utf-8

# # 4. Features and algorithm to maximize the Poi identification

# In[1]:


#coding: utf-8 

### Import some packages and modules to be used later
import pickle
import pandas as pd
import numpy as np

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#from PML_1_data_adq import data_dict_cor 

from sklearn import datasets

from feature_format import featureFormat, targetFeatureSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, SelectFdr

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import cross_val_score


from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, precision_score, recall_score 
from sklearn.metrics import classification_report, f1_score, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC, SVC


# # 4.0. Open file with scaled features (from PML_3) and define features lists

# In[2]:


data_dict_rob=pickle.load(open("data_dict_rob.pkl", "rb"))
data_df=pd.DataFrame.from_dict(data_dict_rob, orient='index')

data_df.info()


# We need to specify the origina feature list and the ones that integrate the new features  

# In[3]:


# 1. original features
features_list1=[ 'poi',
                               'salary', 'bonus', 'long_term_incentive',
                               #'deferred_income',#'deferral_payments', #'loan_advances',
                               'other', 'expenses',#'director_fees',  
                               'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               #'restricted_stock_deferred', 
                               'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi']

# 2. features including new after create_feat
features_new=[ "incentives","total_money",
                'incentive_f','payment_f','payment_2','payment_tt','emails_exc',
                'incentives_ratio',"salary_ratio", "bonus_ratio", 
                "expenses_ratio","other_ratio",  
                "exercised_stock_options_ratio", 
                "total_payments_ratio", "total_stock_value_ratio", 
                "from_poi_to_this_person_ratio", "from_this_person_to_poi_ratio"]

features_list_new=list(set(features_list1+features_new))

# 3. features list without poi . The one to pass to the algo
poi=['poi']
algo_list= list(set(features_list_new).difference(set(poi)))

# 4. features list with poi
features_list=algo_list.copy()
features_list.insert(0,'poi')

# 5. output list with poi and poi_pred
output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# 6. orgianl features list without poi, with poi and poi_pred
algo_list0= list(set(features_list).difference(set(poi)))
algo_list0= list(set(algo_list0).difference(set(features_new)))
output_list0=features_list1.copy()
output_list0.insert(1,'poi_pred')


# # 4.1. Comparing classifiers with different features
# 
# The objective then is try to find an algorithm that in combination with some features provides a precision and recall are both at least 0.3
# My particular objetive is to find the clasifier that answer better to our questions: who are poi?.
# That measn we are going to privilige the identificationof poi out of poi than the identification of non poi out of non poi
# That means we prigilege false positives than false negatives 
# Lets see what are the result with different classfier using all original features

# ## 4.1.1. Comparing classifiers with original features

# In[5]:


# Comparing clasifier with original features w/o big Nan features and after robusscaler

#train and test sets
f_list=algo_list0
set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
features_test = set_test[f_list ]
labels_test=set_test.poi

#Classifiers
clf1 = LogisticRegression(random_state=1)
clf2 = GaussianNB()
clf3 = AdaBoostClassifier()
clf4 = DecisionTreeClassifier()
clf5 = DecisionTreeClassifier(random_state=0)
clf6 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
clf7 = DecisionTreeClassifier(max_depth=None, min_samples_split=5,random_state=42)
clf8 = RandomForestClassifier()
clf9 = RandomForestClassifier(random_state=0)
clf10 = RandomForestClassifier(random_state=42)
clf11 = RandomForestClassifier(n_estimators=50, random_state=42)
clf12 = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
clf13 =  ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=42)

#voting classifier
eclf = VotingClassifier(
    estimators=[('Logistic Regression', clf1), ('naive Bayes', clf2), ('Ada', clf3), 
                ('DecTre', clf4), ('DecTre0', clf5), ('DecTre20', clf6), ('DecTre542', clf7),
                ('RF', clf8), ('RF0', clf9), ('RF42', clf10), ('RF5042', clf11), ('RF5020', clf12), 
                ('ET10242', clf13)],
    voting='hard')

#classifier iteration and scores
for clf, label in zip([clf1, clf2, clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10,clf11,
                       clf12,clf13,eclf], ['Logistic Regression','naive Bayes',  'Ada',  'RF','RF0',
                                           'RF42', 'RF5042' , 'RF5020', 'ET10242', 
                                           'Ensemble']):
    scores = cross_val_score(clf, features_train, labels_train, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    fit = clf.fit(features_train, labels_train)
    dt_pred = clf.predict(features_test)
    labels_pred=clf.predict(features_test)

    print("CONFUSION MATRIX","\n",confusion_matrix(labels_test,labels_pred))
    dt_precision = precision_score(labels_test, dt_pred)
    dt_recall = recall_score(labels_test, dt_pred)
    print("PRECISSION",dt_precision, "RECALL",dt_recall,"\n")


# Best one are Logistic and RandomForest basic and with different parameters.
# There are Decission Tree classifier that identified 3 poi out of 4 but their error in idientified non poi is too big to use them (>10 out of 40. They identifies several potential poi in the non poi population, we will comment on that later.
# Lets create an ensemble with the best classifier

# In[6]:


# Comparing clasifier with original features w/o big Nan features and after robusscaler

'''f_list=algo_list0

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
   
features_test = set_test[f_list ]
labels_test=set_test.poi'''

clf1 = LogisticRegression(random_state=1)
clf12 = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
clf13 =  ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=42)

eclf = VotingClassifier(
    estimators=[('Logistic Regression', clf1), ('RF5020', clf12), 
                ('ET10242', clf13)],
    voting='hard')

for clf, label in zip([clf1, clf12,clf13,eclf], ['Logistic Regression','RF5020', 'ET10242', 
                                           'Ensemble']):
    scores = cross_val_score(clf, features_train, labels_train, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    fit = clf.fit(features_train, labels_train)
    dt_pred = clf.predict(features_test)
    labels_pred=clf.predict(features_test)

    print("\n","CONFUSION MATRIX","\n",confusion_matrix(labels_test,labels_pred),"\n")
    dt_precision = precision_score(labels_test, dt_pred)
    dt_recall = recall_score(labels_test, dt_pred)
    print("PRECISSION","\n",dt_precision)
    print("RECALL","\n",dt_recall)
    print("\n")


# We do not get our objective of 0.3 in precission and recall, the lets try including the created features

# ## 4.1.2. Comparing classifiers with original and created features

# In[7]:


# 1. comparing clasifier with 32 features after robustscaler

f_list_all=algo_list 
set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train_a = set_train[f_list_all]
labels_train_a=set_train.poi
   
features_test_a= set_test[f_list_all]
labels_test_a=set_test.poi

clf1 = LogisticRegression(random_state=1)
clf2 = GaussianNB()
clf3 = AdaBoostClassifier()
clf4 = DecisionTreeClassifier()
clf5 = DecisionTreeClassifier(random_state=0)
clf6 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
clf7 = DecisionTreeClassifier(max_depth=None, min_samples_split=5,random_state=42)
clf8 = RandomForestClassifier()
clf9 = RandomForestClassifier(random_state=0)
clf10 = RandomForestClassifier(random_state=42)
clf11 = RandomForestClassifier(n_estimators=50, random_state=1)
clf12 = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
clf13 =  ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=42)

eclf = VotingClassifier(
    estimators=[('Logistic Regression', clf1), ('naive Bayes', clf2), ('Ada', clf3), 
                ('DecTre', clf4), ('DecTre0', clf5), ('DecTre20', clf6), ('DecTre542', clf7),
                ('RF', clf8), ('RF0', clf9), ('RF42', clf10), ('RF501', clf11), ('RF5020', clf12), 
                ('ET10242', clf13)],
    voting='hard')

for clf, label in zip([clf1, clf2, clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10,clf11,
                       clf12,clf13,eclf], ['Logistic Regression','naive Bayes',  'Ada',
                                           'DecTre' , 'DecTre0', 'DecTre20','DecTre542',
                                            'RF','RF0', 'RF42', 'RF501' , 'RF5020', 'ET10242', 
                                           'Ensemble']):
    scores = cross_val_score(clf, features_train_a, labels_train_a, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    fit = clf.fit(features_train_a, labels_train_a)
    dt_pred_a = clf.predict(features_test_a)
    labels_pred_a=clf.predict(features_test_a)

    print("CONFUSION MATRIX","\n",confusion_matrix(labels_test_a,labels_pred_a))
    dt_precision = precision_score(labels_test_a, dt_pred_a)
    dt_recall = recall_score(labels_test_a, dt_pred_a)
    print("PRECISSION",dt_precision, "RECALL",dt_recall)
    print("\n")    
    


# Best one are Ada and RandomForest basic and with different parameters.
# There are Decission Tree classifier that identified 2 poi out of 4 but their error in idientified non poi is too big to use them (>5 out of 40).
# Lets create an ensemble with the best classifier

# In[8]:


# 1. comparing clasifier with 32 features after robustscaler

f_list=algo_list 
set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
   
features_test = set_test[f_list ]
labels_test=set_test.poi

clf3 = AdaBoostClassifier()
clf10 = RandomForestClassifier(random_state=42)
clf13 =  ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=42)


eclf = VotingClassifier(
    estimators=[ ('Ada', clf3), 
                ('RF42', clf10),
                ('ET10242', clf13)],
    voting='hard')

for clf, label in zip([clf3,clf10,clf13,
                       eclf], [ 'Ada','RF42', 'ET10242', 
                                           'Ensemble']):
    scores = cross_val_score(clf, features_train_a, labels_train_a, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    fit = clf.fit(features_train_a, labels_train_a)
    dt_pred_a = clf.predict(features_test_a)
    labels_pred_a=clf.predict(features_test_a)

    print("\n","CONFUSION MATRIX","\n",confusion_matrix(labels_test_a,labels_pred_a),"\n")
    dt_precision = precision_score(labels_test_a, dt_pred_a)
    dt_recall = recall_score(labels_test_a, dt_pred_a)
    print("PRECISSION","\n",dt_precision)
    print("RECALL","\n",dt_recall)
    print("\n")


# The best one seams Ada with 3 of 4 poi identified and 0.86 accuracy, the false postive are 3 of 40 non poi and Random Forest with 2 of 4 poi identified and 0.85 accuracy, the false postive are 2 of 40 non poi

# # 4.2. Function for algorithm
# We define a function to create the training and test set, to pass the clasifier and to get the result, including the names of the poi and non poi identified as poi 

# In[4]:


#Funtion to pass the algorithm and generate the measurement and the result of the algo

def pass_clf(algo_list, features_list,output_list,data_df,clf):
    
    # features_list includes poi 
    # algo_list is feature list without poi
    # output_list includes poi and poi_pred 
    #train and test set
    set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
    features_train = set_train[algo_list]
    labels_train=set_train.poi
   
    features_test = set_test[algo_list ]
    labels_test=set_test.poi
          
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
    
    #fit and predict the algo
    fit = clf.fit(features_train, labels_train)
    dt_pred = clf.predict(features_test)
    labels_pred=clf.predict(features_test)
    
    #score of algo  
    dt_score = clf.score(features_test, labels_test)
    dt_precision = precision_score(labels_test, dt_pred)
    dt_recall = recall_score(labels_test, dt_pred)
    importance = clf.feature_importances_
    rf_pred = clf.predict(features_test)

    # creation of dataframe with prediction poi/nonpoi
    # 7. output list with poi and poi_pred
   
    df_pred_final = features_test.copy()
    df_pred_final ['poi'] = data_df['poi']
    df_pred_final['poi_pred'] = rf_pred
    df_pred_final= df_pred_final.loc[ :,(output_list)] 
    
    # print of information
    print('data frame shape',data_df.shape)
    print('train frame shape',features_train.shape, labels_train.shape)
    print('test frame shape',features_test.shape, labels_test.shape)
    print("\n")
    
    print("\n","CONFUSION MATRIX","\n",confusion_matrix(labels_test,labels_pred),"\n")
    print("\n")

    print("CLASSIFICATION REPORT","\n",classification_report(labels_test, labels_pred))
    print("\n")
    
    print("PRECISSION","\n",dt_precision)
    print("\n")
    
    print("RECALL","\n",dt_recall)
    print("\n")
    
    # print Features importance
    
    fi_df=pd.DataFrame({'feature':list(algo_list),'importance':clf.feature_importances_}
                  ).sort_values('importance',ascending=False )
    print(fi_df)
    
    
    # print dataframe with poi and poi prediction   
    data_poi = df_pred_final[df_pred_final['poi'] ==True]
    
    pred_poi = df_pred_final[df_pred_final['poi_pred'] ==True]
      
    result_poi = data_poi.append(pred_poi)
    

    return result_poi   


# We will use this funcition later to compare the different classifier fine tuned

# # 4.3. Random forest and original features

# ## 4.3.1. Random forest RState=42 and original features
# 
# Lets select the best features convination for random forest with random state=42 and later fine tune the algorithm
# If we do not get good result we will do that again with other random state value but we need to use a random state value to ensure reproductibiility of the test. Lets start with the perfect number 42

# In[141]:


clf_rf=RandomForestClassifier(random_state=42)

pass_clf(algo_list0, features_list1, output_list0, data_df, clf_rf)


# This is a good result, 1 poi out of 4 poi and only 1 non poi out of 40 identified as poi
# The algorithm identifies poi COLWELL WESLEY, miss the poi LAY KENNET, H LHANNON KEVIN P and KOENIG MARK E
# And identified LAVORATO JOHN J as poi when they are considered non poi
# 
# Lets see if we use only some feature we get better result.
# We will use RFECV to identify how many features we could use to maximize the result

# In[75]:


f_list=algo_list0

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
   
features_test = set_test[f_list ]
labels_test=set_test.poi

X_train=features_train
y_train= labels_train

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
forest =  RandomForestClassifier(random_state=42)

rfecv = RFECV(estimator=forest, cv=StratifiedKFold(5), scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_)+1 ), rfecv.grid_scores_, '--o')
indices = rfecv.get_support()
columns = X_train.columns[indices]


# RFECV maximun score is over 86% then we are better using all the features
# We can create a pipeline to test the clasifier over the features selected by a selector

# In[71]:


'''f_list=algo_list0

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
   
features_test = set_test[f_list ]
labels_test=set_test.poi

X=features_train
y= labels_train'''


pipe=Pipeline([('select',SelectKBest(k=9)),('rfc', RandomForestClassifier (random_state=42))])

pipe.fit(features_test,labels_test)
pipe.score(features_test,labels_test)


# Using SelectKBest with 9 features we could get 95% acuracy
# We have the features ordered depending on their importancem then we pass the most important to the algorithm
# The importance is a %, then more features we have, less importance each have as 1 is the total amount of importance.
#                     feature  importance
# 10        total_stock_value    0.163088
# 7                  expenses    0.141091
# 5                     bonus    0.104008
# 9       long_term_incentive    0.103548
# 4                     other    0.091039
# 11  exercised_stock_options    0.079954
# 12  from_this_person_to_poi    0.065708
# 2            total_payments    0.061876
# 8   from_poi_to_this_person    0.046845
# 1             from_messages    0.045101
# 0               to_messages    0.035342
# 3          restricted_stock    0.030596
# 6                    salary    0.019653
# 13  shared_receipt_with_poi    0.012152
# 

# In[74]:


algo_list_rf=[ 'total_stock_value','expenses', 'bonus','long_term_incentive',
              'other','exercised_stock_options','from_this_person_to_poi',
             'total_payments','from_poi_to_this_person'
             ]

clf_rf=RandomForestClassifier (random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# In[69]:


algo_list_rf=['bonus','exercised_stock_options', 'total_stock_value', 'other','expenses'
             ]
clf_rf=RandomForestClassifier (random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This is a good result, accuracy 91%, 2 poi out of 4 poi and only 2 non poi out of 40 identified as poi
# The algorithm identifies poi LAY KENNETH L and HANNON KEVIN P and miss the poi COLWELL WESLEY and KOENIG MARK E
# And identified LAVORATO JOHN J and PAI LOU  as poi when they are considered non poi
# 
# Lets see if we use add any other features more it improves.

# In[13]:


algo_list_rf=['bonus','exercised_stock_options', 'total_stock_value', 'other','expenses',
            #'restricted_stock', 'salary', 'long_term_incentive', 'from_messages','from_poi_to_this_person',
             'total_payments', 'shared_receipt_with_poi','to_messages','from_this_person_to_poi']
clf_rf=RandomForestClassifier (random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This is a good result, accuracy 89%, 2 poi out of 4 poi and only 3 non poi out of 40 identified as poi
# The algorithm identifies poi LAY KENNETH L and HANNON KEVIN P and miss the poi COLWELL WESLEY and KOENIG MARK E
# And identified PAI LOU, HICKERSON GARY J and DERRICK JR. JAMES V as poi when they are considered non poi
# 
# Lets see if we use only 5 features if that improves.

# In[14]:


algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
             ]
clf_rf=RandomForestClassifier (random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This is a very good result, 95% accuracy, 3 poi out of 4 poi and only 1 non poi out of 40 identified as poi only by adding total payments features
# The algorithm identifies poi LAY KENNETH L, HANNON KEVIN P and COLWELL WESLEY, and miss the poi KOENIG MARK E
# And identified HICKERSON GARY J as poi when they are considered non poi
# 
# Lets see if we use only some feature we get better result.

# In[15]:


algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi',
              'total_payments', 'long_term_incentive'
             ]

clf_rf=RandomForestClassifier (random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This does not go in the good direction, we have lost accuracy and the identification of poi is 1 out of 3
# LAY KENNETH L is not identified as poi, then this is not good combination
# 
# Lets see if we add one more we get better result.

# In[16]:


algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi',
              'total_payments', 'long_term_incentive', 'from_poi_to_this_person'
    ]

clf_rf=RandomForestClassifier (random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This does not go in the good direction, we have lost accuracy and the identification of poi is 0 out of 4
# LAY KENNETH L is not identified as poi, then this is not good combination
# 
# Lets  add one more 

# In[17]:


algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi',
              'total_payments', 'long_term_incentive', 'from_poi_to_this_person',
                'shared_receipt_with_poi'
             ]


from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier (random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This does not go in the good direction. We have get the maximun with 5 features
# 
# Lets see if using the best cobinations of features and improve by finetuning the algoritm .

# ## 4.3.2. Random forest and GridSearchCV to fine-tune with chosen features 

# In[124]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 900, num = 10)]
# Number of features to consider at every split
#max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
#bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
              #'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               #'bootstrap': bootstrap
              }
print(random_grid)


# In[131]:


# GridSearchCV to find the best finetune for the algorithm with sme result from CV

algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'  ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_rf]
labels_train=set_train.poi
   
features_test = set_test[algo_list_rf]
labels_test=set_test.poi

from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(random_state=42)

param_grid={
            'n_estimators':[200, 400, 600],
            'max_depth':[10, 20, 30, 40, 50, 60],
         #  'max_features':['auto', 'sqrt'],
            'min_samples_split': [2, 5, 10], 
            'min_samples_leaf': [1, 2, 4], 
            'bootstrap': [True, False],
            'random_state':[42]     }
f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_rf,param_grid=param_grid )

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# In[133]:


# algorithm with result of GridSearch and random state=42 


algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
             ]

clf_rf1=RandomForestClassifier(n_estimators=200,  max_depth=10,  
                               min_samples_leaf= 1, min_samples_split= 10, bootstrap= True,random_state=42)


features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf1)


# In[18]:


# algorithm without tune parameters
algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
             ]
clf_rf=RandomForestClassifier (random_state=42)
output_list_rf.insert(1,'poi_pred')


features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This is a very good result, 95% accuracy, 3 poi out of 4 poi and only 1 non poi out of 40 identified as poi only by adding total payments features The algorithm identifies poi LAY KENNETH L, HANNON KEVIN P and COLWELL WESLEY, and miss the poi KOENIG MARK E And identified HICKERSON GARY J as poi when they are considered non poi
# 
# Then we will use the features 'exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
# Lets see if we can fine tune the algorithm to get better.
# Lets use GridSearchCV to test the basic parameters on RandomForest: n_estimator (number of trees to be generated), max_depth *, max_features (max number is the total of features we pass to each tree, the max can not exced the number of features we use for the algorithm)

# In[93]:


# GridSearchCV to find the best finetune for the algorithm

algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'  ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_rf]
labels_train=set_train.poi
   
features_test = set_test[algo_list_rf]
labels_test=set_test.poi

from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(random_state=42)

param_grid={
            'n_estimators': [10,200,500],
            'max_depth':[7,15,24,200,500],
            'max_features':[2,5],
            'random_state':[42]     }
f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_rf,param_grid=param_grid )

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# The tune of the algorithm is needed only for the parametres different to default.
# 
# The result of GridSearch is different depending on the parametres.
# 
# We give some values for the parameters we want to use and we get this result
# { 'max_depth': 7,
#  'max_features': 2,
#  'n_estimators': 200,
#  'random_state': 42}

# In[135]:


# algorithm with result of GridSearch and random state=42 


algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
             ]

clf_rf1=RandomForestClassifier(n_estimators=200,  max_depth=7, max_features = 2, random_state=42)


features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf1)


# In[129]:


# algorithm with result of GridSearch and random state=0 

algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']

clf_rf3=RandomForestClassifier(n_estimators=100,  max_depth=None, max_features = 2,  random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf3)


# This result is not as good as the algorithm with no parameters scept random_state=42 
# 
# Lets change some of the parameters to try to find a better result

# In[126]:


# algorithm with result of GridSearch and random state=42 

algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']

clf_rf4=RandomForestClassifier(n_estimators=10,  max_depth=7, max_features = 2,  random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf4)


# In[121]:


def pass_clf_noimp(algo_list, features_list,output_list,data_df,clf):
    
    # features_list includes poi 
    # algo_list is feature list without poi
    # output_list includes poi and poi_pred 
    #train and test set
    set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
    features_train = set_train[algo_list]
    labels_train=set_train.poi
   
    features_test = set_test[algo_list ]
    labels_test=set_test.poi
          
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
    
    #fit and predict the algo
    fit = clf.fit(features_train, labels_train)
    dt_pred = clf.predict(features_test)
    labels_pred=clf.predict(features_test)
    
    #score of algo  
    dt_score = clf.score(features_test, labels_test)
    dt_precision = precision_score(labels_test, dt_pred)
    dt_recall = recall_score(labels_test, dt_pred)
   # importance = clf.feature_importances_
    rf_pred = clf.predict(features_test)

    # creation of dataframe with prediction poi/nonpoi
    # 7. output list with poi and poi_pred
   
    df_pred_final = features_test.copy()
    df_pred_final ['poi'] = data_df['poi']
    df_pred_final['poi_pred'] = rf_pred
    df_pred_final= df_pred_final.loc[ :,(output_list)] 
    
    # print of information
    print('data frame shape',data_df.shape)
    print('train frame shape',features_train.shape, labels_train.shape)
    print('test frame shape',features_test.shape, labels_test.shape)
    print("\n")
    
    print("\n","CONFUSION MATRIX","\n",confusion_matrix(labels_test,labels_pred),"\n")
    print("\n")

    print("CLASSIFICATION REPORT","\n",classification_report(labels_test, labels_pred))
    print("\n")
    
    print("PRECISSION","\n",dt_precision)
    print("\n")
    
    print("RECALL","\n",dt_recall)
    print("\n")
    
    # print Features importance
    
   # fi_df=pd.DataFrame({'feature':list(algo_list),'importance':clf.feature_importances_}
   #               ).sort_values('importance',ascending=False )
   # print(fi_df)
    
    
    # print dataframe with poi and poi prediction   
    data_poi = df_pred_final[df_pred_final['poi'] ==True]
    
    pred_poi = df_pred_final[df_pred_final['poi_pred'] ==True]
      
    result_poi = data_poi.append(pred_poi)
    
    
    return result_poi   






# algorithm dumy

algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']

from sklearn.dummy import DummyClassifier
clf_dummy = DummyClassifier(strategy='most_frequent', random_state=0)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf_noimp(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_dummy)


# In[23]:


# algorithm with tune n_estimators=100, bootstrap = True, max_features = 'sqrt' parameters

algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
             ]
from sklearn.ensemble import RandomForestClassifier
clf_rf1=RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf1)


# In[24]:


# algorithm with tune n_estimators=50, criterion='entropy', random_state = 42 parameters

from sklearn.ensemble import RandomForestClassifier

clf_rf2=RandomForestClassifier(n_estimators=50, criterion='entropy', random_state = 42)
pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf2)


# In[25]:


# algorithm with tune n_estimators=500, max_features='sqrt',max_depth=None, 
# min_samples_split=2, random_state=42) parameters

from sklearn.ensemble import RandomForestClassifier

clf_rf3=RandomForestClassifier (n_estimators=500, max_features='sqrt',max_depth=None,
                                min_samples_split=2, random_state=42)
pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf3)


# ## 4.3.3. Random forest fine-tune with chosen features 

# After try on finetuning the algorithm a bit more and we get some good result. 
# 
# RandomForestClassifier(n_estimators=10,  max_depth=4, max_features = 2, random_state=42) and
# RandomForestClassifier(n_estimators=10,  max_depth=7, max_features = 2,  random_state=42)
# 
# That means that it does not need to go to depth 7 as 4 gives same result
# 
# Then as conclusion if we want to use only the original feature the best is 
# 
# features: 'exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
# clasifier: =RandomForestClassifier (n_estimators=10,  max_depth=4, max_features = 2, random_state=42)
# 

# As who is non poi is influenced by political and investigation agreement, the algorithm shoud basecaly be able to identify the poi as poi and it is normal that idenfity some considered non poi as poi 
# It will be interesting to investigate the false positive with an algorithm that identify properly poi as poi
# 
# Let see DecisionTreeClassifier(random_state=42)

# In[26]:


#algo_list_rf=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']
algo_list_rf=algo_list0
clf_rf=DecisionTreeClassifier(random_state=42)

features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# It will be interesting to investigate if some of the 11 non poi identified as poi could be considered poi in any way
# WHALLEY LAWRENCE G
# JACKSON CHARLENE R
# STABLER FRANK
# KOENIG MARK E
# GAHN ROBERT S
# DERRICK JR. JAMES V
# KISHKILL JOSEPH G
# HANNON KEVIN P
# WODRASKA JOHN
# COLWELL WESLEY
# MCCLELLAN GEORGE
# LAVORATO JOHN J
# PAI LOU L
# METTS MARK
# but that is not part of this study then continue with the project 
# 
# Now lets find the best features for ADA and try to improve his performace

# # 4.4. ADAbost and new features 

# ## 4.4.1. AdaBost and all new features
# Lets try now ADAbost with the new features

# In[27]:


algo_list_ada=algo_list
clf_ada=AdaBoostClassifier()
pass_clf(algo_list_ada, features_list, output_list, data_df, clf_ada)


# This is a good result, 3 poi out of 4 poi and only 3 non poi out of 40 identified as poi
# The algorithm identifies poi LAY KENNETH L, KOENIG MARK E and COLWELL WESLEY, miss the poi HANNON KEVIN P 
# And identified WHALLEY LAWRENCE G,PIPER GREGORY F  and HICKERSON GARY J as poi when they are considered non poi
# The most important features are total_money, expenses, incentives_ratio,payment_f,'exercised_stock_options'
# 
# Lets see if we use only some feature we get better result.
# We will use RFECV to identify how many features we could use to maximize the result

# In[28]:


f_list=algo_list

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
   
features_test = set_test[f_list ]
labels_test=set_test.poi

X=features_train
y= labels_train

from sklearn.feature_selection import RFECV

clf_ada=AdaBoostClassifier()# (random_state=42)
rfecv = RFECV(estimator=clf_ada, cv=14, scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_)+1 ), rfecv.grid_scores_, '--o')
indices = rfecv.get_support()
columns = X_train.columns[indices]
#print('The most important columns are {}'.format(','.join(columns)))         
#print (f_list)


# RFECV maximun score is 83% then we are better using all the features
# We can create a pipeline to test the clasifier over the features selected by a selector

# In[76]:


f_list=algo_list

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
   
features_test = set_test[f_list ]
labels_test=set_test.poi

X=features_train
y= labels_train


pipe=Pipeline([('select',SelectKBest(k=10)),('Ada', AdaBoostClassifier())])
#f_classif,k=4  SelectFdr
pipe.fit(X,y)
pipe.score(features_test,labels_test)


# Using SelectKBest with 11 features we could get 83% acuracy
# We have the features ordered depending on their importancem then we pass the most important to the algorithm
# The importance is a %, then more features we have, less importance each have as 1 is the total amount of importance.
# 
#                           feature  importance
# 26                    total_money        0.16
# 21                       expenses        0.12
# 5                       payment_f        0.08
# 18               incentives_ratio        0.08
# 30        exercised_stock_options        0.06
# 9                       payment_2        0.06
# 7                           other        0.06
# 14                     incentives        0.06
# 19                 total_payments        0.04
# 0   from_this_person_to_poi_ratio        0.04
# 15        from_this_person_to_poi        0.04
# 4               total_stock_value        0.04
# 16                 expenses_ratio        0.02
# 22            long_term_incentive        0.02
# 6                     other_ratio        0.02
# 17                         salary        0.02
# 20        shared_receipt_with_poi        0.02
# 1                     incentive_f        0.02
# 8                restricted_stock        0.02
# 12                    to_messages        0.02
# 13  exercised_stock_options_ratio        0.00
# 11                    bonus_ratio        0.00
# 3                      emails_exc        0.00
# 23                     payment_tt        0.00
# 24                   salary_ratio        0.00
# 25        from_poi_to_this_person        0.00
# 2                   from_messages        0.00
# 27                          bonus        0.00
# 28        total_stock_value_ratio        0.00
# 29           total_payments_ratio        0.00
# 10  from_poi_to_this_person_ratio        0.00

# In[77]:


algo_list_ada=['total_money','expenses','incentives_ratio','other', 'payment_f',
               'exercised_stock_options', 'from_this_person_to_poi','total_stock_value',
               "incentives", 'payment_2']
             

clf_ada=AdaBoostClassifier()

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# In[30]:


algo_list_ada=['total_money','expenses','payment_f','incentives_ratio', 'exercised_stock_options'
             ]

clf_ada=AdaBoostClassifier()

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# This is a not so good result, accuracy 86%, 2 poi out of 4 poi and 4 non poi out of 40 identified as poi
# The algorithm identifies poi LAY KENNETH L, and COLWELL WESLEY and miss the poi KOENIG MARK E and HANNON KEVIN P  
# And identified DIMICHELE RICHARD G, WHALLEY LAWRENCE G, PIPER GREGORY F and HICKERSON GARY J as poi when they are considered non poi
# 
# Lets see if we use add any other features more it improves.

# In[31]:


algo_list_ada=['total_money','expenses','payment_f','incentives_ratio', 'exercised_stock_options',
              'payment_2', #'other', 'incentives'
             ]


clf_ada=AdaBoostClassifier()

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# This is a good result, accuracy 91%, 3 poi out of 4 poi and only 3 non poi out of 40 identified as poi
# The algorithm identifies poi LAY KENNETH L, KOENIG MARK E and COLWELL WESLEY and miss the poi  HANNON KEVIN P  
# And identified DIMICHELE RICHARD G, WHALLEY LAWRENCE G and HICKERSON GARY J as poi when they are considered non poi
# 
# Lets see if we use add any other features more it improves.

# In[32]:


algo_list_ada=['total_money','expenses',#'payment_f',
              'incentives_ratio', 'exercised_stock_options',
             # 'payment_2', 
              'other','incentives'
             ]

clf_ada=AdaBoostClassifier()

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# This is a good result, accuracy 86%, 2 poi out of 4 poi and 4 non poi out of 40 identified as poi
# That does not help 
# 
# Lets see if we use add any other features more it improves.

# In[33]:


algo_list_ada=['total_money','expenses',#'payment_f',
              'incentives_ratio', 'exercised_stock_options',
              'payment_2', 'other','incentives',
             #  'total_payments',
              'from_this_person_to_poi_ratio',
              'from_this_person_to_poi'        ]

clf_ada=AdaBoostClassifier()

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# This is a good result, accuracy 93%, 3 poi out of 4 poi and 2 non poi out of 40 identified as poi
# 
# The algorithm identifies poi LAY KENNETH L, KOENIG MARK E and COLWELL WESLEY and miss the poi  HANNON KEVIN P  
# And identified KISHKILL JOSEPH G and HICKERSON GARY J as poi when they are considered non poi
# 
# Lets see if we use add any other features more it improves.
# 
# Lets see if we use add any other features more it improves.

# In[84]:


algo_list_ada2=['total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other',
              'from_this_person_to_poi' ]

clf_ada=AdaBoostClassifier()

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada2,features_list_ada, output_list_ada, data_df, clf_ada)


# In[82]:


algo_list_ada=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments' ]

clf_ada=AdaBoostClassifier()

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# This is a same result, accuracy 93%, 3 poi out of 4 poi and 2 non poi out of 40 identified as poi but using less features, only 7.
# 
# The algorithm identifies poi LAY KENNETH L, KOENIG MARK E and COLWELL WESLEY and miss the poi HANNON KEVIN P  
# And identified KISHKILL JOSEPH G and HICKERSON GARY J as poi when they are considered non poi
# 
# Lets see if we can get the same reducing 2 features more.

# In[35]:


algo_list_ada=['total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other',
                          ]


clf_ada=AdaBoostClassifier()

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# This is a same result, accuracy 93%, 3 poi out of 4 poi and 2 non poi out of 40 identified as poi but using less features, only 5.
# 
# The algorithm identifies poi LAY KENNETH L, KOENIG MARK E and COLWELL WESLEY and miss the poi HANNON KEVIN P  
# And identified KISHKILL JOSEPH G and HICKERSON GARY J as poi when they are considered non poi
# 
# Lets see if we can improve that finetunning the algorithm 

# ## 4.4.2. ADAboost and GridSearchCV to fine-tune with chosen features 

# In[17]:


# GridSearchCV to find the best finetune for the algorithm

algo_list_ada=['total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other', 'from_this_person_to_poi' ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_ada]
labels_train=set_train.poi
   
features_test = set_test[algo_list_ada]
labels_test=set_test.poi

clf_ada=AdaBoostClassifier(random_state=42)

param_grid={'n_estimators':[7,15,24,200,500,900],
            'learning_rate':[1, 1.5, 2, 2.5, 3,],
             'random_state':[42]     }

f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_ada,param_grid=param_grid )

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# In[18]:


algo_list_ada=['total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other', 'from_this_person_to_poi'
                          ]
clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=15, learning_rate=1.5, 
                           random_state=42)

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# In[38]:


algo_list_ada=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']   

clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
                          algorithm='SAMME.R', random_state=42)

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# In[14]:


# GridSearchCV to find the best finetune for the algorithm

algo_list_ada=['total_money','expenses','payment_f','incentives_ratio', 'exercised_stock_options',
              'payment_2'
                          ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_ada]
labels_train=set_train.poi
   
features_test = set_test[algo_list_ada]
labels_test=set_test.poi

clf_ada=AdaBoostClassifier(random_state=42)

param_grid={'n_estimators':[7,15,24,200,500,900],
            'learning_rate':[1, 1.5, 2, 2.5, 3,],
            'algorithm':['SAMME', 'SAMME.R'],
             'random_state':[42]     }

f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_ada,param_grid=param_grid )

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# In[15]:


algo_list_ada=['total_money','expenses','payment_f','incentives_ratio', 'exercised_stock_options',
              'payment_2'
                          ] 

#clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
#                          algorithm='SAMME.R', random_state=42)

clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=15, learning_rate=1.0, 
                          algorithm='SAMME', random_state=42)

#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=50, learning_rate=1.5,  random_state=42)
#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=500, learning_rate=1.5,  random_state=42)                           
# clf_ada=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=1, min_samples_split=2,random_state=42) ,n_estimators=500)
                                                     
                           #(max_depth=None, min_samples_split=5,random_state=42))

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# The trail on finetuning the algorithm a bit more has not get better result. Two of them are better in identifing non poi as non poi but as we want to maximize the poi identification, those algorithm are not more pertinet than the previous one.
# 
# Then as conclusion if we want to use only the original feature the best is 
# 
# features: 'total_money','expenses','incentives_ratio','exercised_stock_options', 'other', 'from_this_person_to_poi_ratio', 'from_this_person_to_poi' 
# 
# clasifier:AdaBoostClassifier()

# ADABost finetune to identify non poi as poi that could be suspicious of being poi

# In[39]:


algo_list_ada=['total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other'
                          ]
clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=2.35, 
                          algorithm='SAMME.R', random_state=42)

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# # 4.5. Random forest RState=42 with all features

# ## 4.5.1. Random forest RState=42 and new features to chose

# In[41]:


algo_list_rf_all=[ 'salary', 'bonus', 'long_term_incentive',
    #'deferred_income',#'deferral_payments', #'loan_advances',
    'other', 'expenses',#'director_fees',  
    'total_payments','exercised_stock_options', 'restricted_stock',
                               #'restricted_stock_deferred', 
                               'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi',
                               'shared_receipt_with_poi',
  "incentives","total_money",
                'incentive_f','payment_f','payment_2','payment_tt','emails_exc',
                'incentives_ratio',"salary_ratio", "bonus_ratio", 
                "expenses_ratio","other_ratio",  
                "exercised_stock_options_ratio", 
                "total_payments_ratio", "total_stock_value_ratio", 
                "from_poi_to_this_person_ratio", "from_this_person_to_poi_ratio"]

clf_rf_all=RandomForestClassifier (random_state=42)
pass_clf(algo_list_rf_all, features_list, output_list, data_df, clf_rf_all)


# This is a realy bad result, 0 poi out of 4 poi. The accuracy is 89
# 
# 
# Lets see if we use only some feature we get better result.
# We will use RFECV to identify how many features we could use to maximize the result

# In[42]:


f_list=algo_list_rf_all

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
   
features_test = set_test[f_list ]
labels_test=set_test.poi

X_train=features_train
y_train= labels_train

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
forest =  RandomForestClassifier(random_state=42)
rfecv = RFECV(estimator=forest, cv=StratifiedKFold(5), scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_)+1 ), rfecv.grid_scores_, '--o')
indices = rfecv.get_support()
columns = X_train.columns[indices]
#print('The most important columns are {}'.format(','.join(columns)))         
#print (f_list)


# RFECV maximun score is 88% then we are better using all the features
# We can create a pipeline to test the clasifier over the features selected by a selector

# In[43]:


f_list=algo_list_rf_all

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[f_list]
labels_train=set_train.poi
   
features_test = set_test[f_list ]
labels_test=set_test.poi

X=features_train
y= labels_train


pipe=Pipeline([('select',SelectKBest(k=5)),('rfc', RandomForestClassifier (random_state=42))])
#f_classif,k=4  SelectFdr
pipe.fit(X,y)
pipe.score(features_test,labels_test)


# Using SelectKBest with 17 features we could get 91% acuracy
# We have the features ordered depending on their importancem then we pass the most important to the algorithm
# The importance is a %, then more features we have, less importance each have as 1 is the total amount of importance.
# 
#                           feature  importance
# 18        exercised_stock_options    0.106106
# 19                     payment_tt    0.100241
# 5                  total_payments    0.086785
# 24                      payment_f    0.071138
# 0               total_stock_value    0.063925
# 1                        expenses    0.057639
# 15                          bonus    0.048143
# 8                incentives_ratio    0.043787
# 16                    bonus_ratio    0.033141
# 28        shared_receipt_with_poi    0.032117
# 2                restricted_stock    0.031559
# 3   exercised_stock_options_ratio    0.029442
# 13                          other    0.025643
# 12                    other_ratio    0.023279
# 11                      payment_2    0.023057
# 21                    incentive_f    0.022786
# 9             long_term_incentive    0.022670
# 17                     emails_exc    0.022148
# 27           total_payments_ratio    0.021980
# 26  from_poi_to_this_person_ratio    0.019199
# 30                    to_messages    0.018408
# 20        total_stock_value_ratio    0.017192
# 4                    salary_ratio    0.017146
# 7                     total_money    0.011930
# 6         from_this_person_to_poi    0.011278
# 10                         salary    0.011065
# 29                 expenses_ratio    0.010613
# 14                  from_messages    0.007244
# 23                     incentives    0.005348
# 22        from_poi_to_this_person    0.004991
# 25  from_this_person_to_poi_ratio    0.000000
# 
# 
# Lets check adding one by one to get progresively results

# In[44]:


algo_list_rf=['exercised_stock_options', 
              'payment_tt',
             'incentive_f','payment_f',
               'total_stock_value',
                'shared_receipt_with_poi', #3, 3
            'bonus',"total_money"
              ,'expenses',
               "bonus_ratio", 'total_payments'
              ,"other_ratio" ,'other',
              'restricted_stock','payment_2'#4,2
              ,'exercised_stock_options',
               "incentives"
              ,'long_term_incentive',
              'salary', 'total_stock_value_ratio'
              , 'to_messages',"salary_ratio", 'from_this_person_to_poi'
   ,'emails_exc',  
   'total_payments_ratio',
                               'from_poi_to_this_person' #2,2
              , 'from_this_person_to_poi' ,"expenses_ratio" , 'incentives_ratio',
               'from_messages',  'from_poi_to_this_person'
             ]

from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(#n_estimators=10, max_depth=7,max_features=2, 
                                 random_state=42)


features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# In[45]:


algo_list_rf=['exercised_stock_options', 'total_payments',
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  ]
    
from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2, 
                                 random_state=42)


features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This is a very good result, accuracy 95%, 3 poi out of 4 poi and 1 non poi out of 40 identified as poi using 6 features.
# 
# The algorithm identifies poi LAY KENNETH L, COLWELL WESLEY and HANNON KEVIN P and miss the poi KOENIG MARK E   
# And identified DERRICK JR. JAMES V  as poi when he is considered non poi
# 
# I like more this algorithm because he identify as poi 3 considered non poi but that got extrabonus in the middle of the crisis
# HICKERSON GARY J 700k  
# LAVORATO JOHN J 5m
# PAI LOU L who resigned frm the compay six months before the scandal, was questioned during the investigation but not charged
# WHALLEY LAWRENCE G
# 

# 0  exercised_stock_options         NaN
# 1                    other         NaN
# 2                    bonus         NaN
# 3  from_this_person_to_poi         NaN
# 4           total_payments         NaN
# 
# 
#    bonus    0.243822
# 3                 expenses    0.211445
# 0  exercised_stock_options    0.199058
# 2        total_stock_value    0.182807
# 5         incentives_ratio    0.087966
# 1           total_payments    0.074903
# 
# exercised_stock_options
# total_payments 
# other    
# bonus   
# from_this_person_to_poi 
# expenses
# total_stock_value
# incentives_ratio

# ## 4.5.2. Random forest finetune to choose features 

# In[99]:


algo_list_rf=['exercised_stock_options', 
              'payment_tt',
              'total_payments',  
              'payment_f',  
              'total_stock_value',
              'expenses'
              ,'bonus'
              ,'incentives_ratio'
              ,'bonus_ratio'
              ,'shared_receipt_with_poi'
              ,'restricted_stock'
              ,'exercised_stock_options_ratio'
              ,'other','other_ratio','payment_2','incentive_f','long_term_incentive'
              ,'emails_exc' ,'total_payments_ratio', 'from_poi_to_this_person_ratio','to_messages'
              ,'total_stock_value_ratio', 'salary_ratio','total_money','from_this_person_to_poi'
              , 'salary', 'expenses_ratio'
             ]

from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2, 
                                 random_state=42)


features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# This is a good result, accuracy 95%, 2 poi out of 4 poi and all non poi identified.
# The point is that we want to find an algorithm that identify poi better than one that identify non poi, then this combination is not in line with our objective.
# 
# Lets see if we can improve that finetunning the algorithm 

# In[100]:


algo_list_rf=['exercised_stock_options', 'payment_tt', 'total_payments', 'payment_f',  
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  ]

from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2, 
                                 random_state=42)


features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# In[101]:


algo_list_rf=['exercised_stock_options', 'total_payments', 'other'  ,'bonus',
              'from_this_person_to_poi',  'expenses',
              'total_stock_value', 'incentives_ratio'  ]

from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2, 
                                 random_state=42)


features_list_rf=algo_list_rf.copy()
features_list_rf.insert(0,'poi')

output_list_rf=features_list_rf.copy()
output_list_rf.insert(1,'poi_pred')

pass_clf(algo_list_rf, features_list_rf, output_list_rf, data_df, clf_rf)


# Lets ensemble AdaBoost and Random Forest

# # 4.6. AdaBoost with randon forest

# ## 4.6.1. AdaBoost with Randon forest: original features

# In[40]:


# algorithm AdaBoost over the best of RandomForest
#clf_rf4 = RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2,random_state=42)


algo_list_rf_ada=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
                  ]
clf_ada=AdaBoostClassifier((RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2, random_state=42)), 
                           n_estimators=900, learning_rate=2.0, algorithm='SAMME.R', random_state=42)

features_list_rf_ada=algo_list_rf_ada.copy()
features_list_rf_ada.insert(0,'poi')

output_list_rf_ada=features_list_rf_ada.copy()
output_list_rf_ada.insert(1,'poi_pred')




pass_clf(algo_list_rf_ada,features_list_rf_ada, output_list_rf_ada, data_df, clf_ada)


# ## 4.6.2. AdaBoost with Randon forest: all features

# In[102]:


# algorithm AdaBoost over the best of RandomForest
#clf_rf4 = RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2,random_state=42)


algo_list_rf_ada=['exercised_stock_options', 'payment_tt', 'total_payments', 'payment_f',  
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  
                 ]

clf_ada=AdaBoostClassifier((RandomForestClassifier(n_estimators=10, max_depth=4,max_features=2, 
                                 random_state=42)), 
                           n_estimators=10, learning_rate=2.0, random_state=42)

features_list_rf_ada=algo_list_rf_ada.copy()
features_list_rf_ada.insert(0,'poi')

output_list_rf_ada=features_list_rf_ada.copy()
output_list_rf_ada.insert(1,'poi_pred')




pass_clf(algo_list_rf_ada,features_list_rf_ada, output_list_rf_ada, data_df, clf_ada)




# In[ ]:


#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=50, learning_rate=1.5,  random_state=42)
#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=500, learning_rate=1.5,  random_state=42)                           
# clf_ada=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=1, min_samples_split=2,random_state=42) ,n_estimators=500)
                                                     
                           #(max_depth=None, min_samples_split=5,random_state=42))


# In[103]:


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators='warn',
                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
                        warm_start=False),


# This is a good result, accuracy 89%, 3 poi out of 4 poi and 4 non poi out of 40 identified as poi using 8 features.
# 
# The algorithm identifies poi LAY KENNETH L, COLWELL WESLEY and HANNON KEVIN P and miss the poi KOENIG MARK E    
# And identified WHALLEY LAWRENCE G, HICKERSON GARY J, LAVORATO JOHN J and PAI LOU L as poi when they are considered non poi
# 
# After those different combinations we have some combinations we can test with different train and test set

# In[104]:


algo_list_rf=['exercised_stock_options', 'payment_tt', 'total_payments', 'payment_f',  
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_rf]
labels_train=set_train.poi
   
features_test = set_test[algo_list_rf]
labels_test=set_test.poi

from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(#n_estimators=10, max_depth=7,max_features=2, random_state=42#
                                )
param_grid={
            'n_estimators': [10,100],
            'max_features':['auto', 'sqrt','log2'],
            'max_depth':[4,5,6,7,24]
           #, 'criterion':['gini','entropy']
           }
f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_rf,param_grid=param_grid, cv=5,scoring=f1)

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# In[105]:


algo_list_rf=['exercised_stock_options', 'payment_tt', 'total_payments', 'payment_f',  
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_rf]
labels_train=set_train.poi
   
features_test = set_test[algo_list_rf]
labels_test=set_test.poi

from sklearn.ensemble import RandomForestClassifier
clf_rf=AdaBoostClassifier((RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2, 
                                 random_state=42)), 
                           n_estimators=10, learning_rate=2.0, random_state=42)
param_grid={
            'n_estimators': [10,100],
           
           }
f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_rf,param_grid=param_grid, cv=5,scoring=f1)

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# In[106]:


# GridSearchCV to find the best finetune for the algorithm

algo_list_ada=['exercised_stock_options', 'payment_tt', 'total_payments', 'payment_f',  
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  
                          ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_ada]
labels_train=set_train.poi
   
features_test = set_test[algo_list_ada]
labels_test=set_test.poi

clf_ada=AdaBoostClassifier((RandomForestClassifier(n_estimators=10, max_depth=5,max_features='sqrt', 
                                random_state=42)),random_state=42)

param_grid={
            'n_estimators':[7,15,24,200,500,900],
            'learning_rate':[1, 1.5, 2, 2.5, 3,],
            #'algorithm':['SAMME', 'SAMME.R'],
                }

f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_ada,param_grid=param_grid )

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# In[107]:


# algorithm AdaBoost over the best of RandomForest
#clf_rf4 = RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2,random_state=42)


algo_list_rf_ada=['exercised_stock_options', 'payment_tt', 'total_payments', 'payment_f',  
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  
                 ]

clf_ada=AdaBoostClassifier((RandomForestClassifier(n_estimators=10, max_depth=5,max_features='sqrt', 
                                random_state=42)), 
                           n_estimators=10, learning_rate=2.0, random_state=42)

features_list_rf_ada=algo_list_rf_ada.copy()
features_list_rf_ada.insert(0,'poi')

output_list_rf_ada=features_list_rf_ada.copy()
output_list_rf_ada.insert(1,'poi_pred')




pass_clf(algo_list_rf_ada,features_list_rf_ada, output_list_rf_ada, data_df, clf_ada)




# ## 4.6.3. ADAboost with Random forest and GridSearchCV to tune with chosen features 

# In[10]:


# GridSearchCV to find the best finetune for the algorithm

algo_list_ada=['total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other'
                          ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_ada]
labels_train=set_train.poi
   
features_test = set_test[algo_list_ada]
labels_test=set_test.poi

clf_ada=AdaBoostClassifier(random_state=42)

param_grid={
            'base_estimator': [RandomForestClassifier(random_state=42), 
               # RandomForestClassifier(n_estimators=10,  max_depth=4, max_features = 2, random_state=42)
                              ],
            'n_estimators':[7,15,24,200,500,900],
            'learning_rate':[1, 1.5, 2, 2.5, 3,],
            'algorithm':['SAMME', 'SAMME.R'],
             'random_state':[42]     }

f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_ada,param_grid=param_grid )

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# In[97]:


# GridSearchCV to find the best finetune for the algorithm

algo_list_ada=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
                          ]

set_train, set_test = train_test_split(data_df, test_size = 0.3, random_state=42)
features_train = set_train[algo_list_ada]
labels_train=set_train.poi
   
features_test = set_test[algo_list_ada]
labels_test=set_test.poi

clf_ada=AdaBoostClassifier(random_state=42)
clf_rf_ada=RandomForestClassifier(n_estimators=200,  max_depth=7, max_features = 2, random_state=42)

param_grid={
            'base_estimator': [clf_rf_ada, 
            
                              ],
            'n_estimators':[7,15,24,200,500,900],
            'learning_rate':[1, 1.5, 2, 2.5, 3,],
            'algorithm':['SAMME', 'SAMME.R'],
             'random_state':[42]     }

f1=make_scorer(f1_score, average='macro')
CV_clf_rf=GridSearchCV(estimator=clf_ada,param_grid=param_grid )

CV_clf_rf.fit(features_train, labels_train)
CV_clf_rf.best_params_


# In[98]:


algo_list_ada=['exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments'
                          ]

#clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
#                          algorithm='SAMME.R', random_state=42)

clf_ada=AdaBoostClassifier((RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=7, max_features=2, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=200,
                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
                        warm_start=False)),
                            n_estimators=7, learning_rate=1.0, 
                          algorithm='SAMME', random_state=42)

#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=50, learning_rate=1.5,  random_state=42)
#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=500, learning_rate=1.5,  random_state=42)                           
# clf_ada=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=1, min_samples_split=2,random_state=42) ,n_estimators=500)
                                                     
                           #(max_depth=None, min_samples_split=5,random_state=42))
        
        
#clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
#                          algorithm='SAMME.R', random_state=42)

#clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0,  algorithm='SAMME.R', random_state=42)

#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=50, learning_rate=1.5,  random_state=42)
#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=500, learning_rate=1.5,  random_state=42)                           
# clf_ada=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=1, min_samples_split=2,random_state=42) ,n_estimators=500)
                                                     
                           #(max_depth=None, min_samples_split=5,random_state=42))

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# In[37]:


algo_list_ada=['total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other'
                          ]

#clf_ada=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
#                          algorithm='SAMME.R', random_state=42)

clf_ada=AdaBoostClassifier((RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators='warn',
                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
                        warm_start=False)),
                            n_estimators=15, learning_rate=1.0, 
                          algorithm='SAMME.R', random_state=42)

#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=50, learning_rate=1.5,  random_state=42)
#clf_ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)
#                           ,n_estimators=500, learning_rate=1.5,  random_state=42)                           
# clf_ada=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=1, min_samples_split=2,random_state=42) ,n_estimators=500)
                                                     
                           #(max_depth=None, min_samples_split=5,random_state=42))

features_list_ada=algo_list_ada.copy()
features_list_ada.insert(0,'poi')

output_list_ada=features_list_ada.copy()
output_list_ada.insert(1,'poi_pred')

pass_clf(algo_list_ada,features_list_ada, output_list_ada, data_df, clf_ada)


# # 4.7 Algorithm and features chosen for tester
# 

# After all these features selection, algorithm selection and tune and trainig and test, I need to choose the ones to submit 
# Those below are the features and algorithms I will test in tester.py to chose the two to finaly submit
# 

# In[6]:


features_rf1=['poi','bonus','exercised_stock_options', 'total_stock_value', 'other','expenses']
features_rf2=['poi','exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']
features_rf3=['poi','bonus','exercised_stock_options', 'total_stock_value', 'other','expenses',
               'total_payments', 'shared_receipt_with_poi','to_messages','from_this_person_to_poi']
features_rf4=['poi','exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']

features_rf5=['poi','exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']
features_rf6=['poi','exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']
features_rf7=['poi','exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments']
features_rf8=['poi','exercised_stock_options', 'payment_tt', 'total_payments', 'payment_f',  
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  ]
features_rf9=['poi','exercised_stock_options', 'total_payments',   
              'total_stock_value', 'expenses' ,'bonus','incentives_ratio'  ]

features_rf10=['poi','total_money','expenses','incentives_ratio','exercised_stock_options', 'other']


features_ada1=['poi','total_money','expenses','incentives_ratio','other', 'payment_f',
               'exercised_stock_options', 'from_this_person_to_poi','total_stock_value',
               "incentives", 'payment_2']
features_ada2=['poi','total_money','expenses','incentives_ratio','other', 
               'exercised_stock_options','from_this_person_to_poi' ,
               'incentives', 'payment_2',
               'from_this_person_to_poi_ratio']
features_ada3=['poi','total_money','expenses','incentives_ratio','other',
               'exercised_stock_options', 'from_this_person_to_poi' ]
features_ada4=['poi','total_money','expenses','incentives_ratio','other',
               'exercised_stock_options' ]
features_ada5=['poi','total_money','expenses','incentives_ratio','other',
              'exercised_stock_options', 'from_this_person_to_poi']
features_ada6=['poi','total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other'
                          ]


features_ada_rf1=['poi','exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments' ]

features_ada_rf2=['poi','exercised_stock_options','other','bonus','from_this_person_to_poi','total_payments' ]
features_ada_rf3=['poi','total_money','expenses','incentives_ratio',
              'exercised_stock_options', 'other'
                          ]

clf_rf1=RandomForestClassifier (random_state=42)
clf_rf2=RandomForestClassifier (random_state=42)
clf_rf3=RandomForestClassifier (random_state=42)
clf_rf4=RandomForestClassifier(n_estimators=10,  max_depth=7, max_features = 2,  random_state=42)
clf_rf5=RandomForestClassifier(n_estimators=200,  max_depth=7, max_features = 2, random_state=42)
clf_rf6=RandomForestClassifier(n_estimators=100,  max_depth=None, max_features = 2,  random_state=42)
clf_rf7=RandomForestClassifier (n_estimators=500, max_features='sqrt',max_depth=None,
                                min_samples_split=2, random_state=42)
clf_rf8=RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2, random_state=42)
clf_rf9=RandomForestClassifier(n_estimators=10,  max_depth=7, max_features = 2,  random_state=42)
clf_rf10=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators='warn',
                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
                        warm_start=False)




clf_ada1=AdaBoostClassifier()
clf_ada2=AdaBoostClassifier()
clf_ada3=AdaBoostClassifier()
clf_ada4=AdaBoostClassifier()
clf_ada5=AdaBoostClassifier(base_estimator=None, n_estimators=15, learning_rate=1.5, random_state=42)
clf_ada6=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
                         algorithm='SAMME.R', random_state=42)

clf_ada_rf1=AdaBoostClassifier((RandomForestClassifier(n_estimators=10, max_depth=7,max_features=2, random_state=42)), 
                           n_estimators=900, learning_rate=2.0, algorithm='SAMME.R', random_state=42)

clf_ada_rf2=AdaBoostClassifier((RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=7, max_features=2, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=200,
                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
                        warm_start=False)),
                            n_estimators=7, learning_rate=1.0, 
                          algorithm='SAMME', random_state=42)

features_ada_rf3=AdaBoostClassifier((RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators='warn',
                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
                        warm_start=False)),
                            n_estimators=50, learning_rate=1.0, 
                          algorithm='SAMME.R', random_state=42)


# Once all those checked I will finetune again to get the best result on tester and submit 2 good examples.
# After passing them to tester, and finetune over tester, I have some that cover the requirement of 
# Precision: > 0.30	Recall: >0.30	 

# 1. test_classifier(clf_rf2, data_dict_rob, features_rfc, folds=1000)
# clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42)
# features_rfc=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio' ]
# 
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=7, max_features=2, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=200,
#                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
#                        warm_start=False)
# 
# Accuracy: 0.88933	Precision: 0.69451	Recall: 0.30350	F1: 0.42241	F2: 0.34201
# 	Total predictions: 15000	True positives:  607	False positives:  267	
#                                 False negatives: 1393	True negatives: 12733
#                         
# 
# 
# 
# 2. test_classifier(clf_rf2, data_dict_rob, features_rfd, folds=1000)
# clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42)
# features_rfd=['poi', 'bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
#               'total_payments','incentives_ratio'  ]
# 
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=7, max_features=2, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=200,
#                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
#                        warm_start=False)
# 
# Accuracy: 0.87967	Precision: 0.59568	Recall: 0.30350	F1: 0.40212	F2: 0.33651
# 	Total predictions: 15000	True positives:  607	False positives:  412	
#                                 False negatives: 1393	True negatives: 12588
# 
# 
# 
# 3. test_classifier(clf_ada2, data_dict_rob, features_adag, folds=1000)
# clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
# features_adag=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio',
#                'from_this_person_to_poi' ]
# 
# AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.5,
#                    n_estimators=500, random_state=42)
# 
# Accuracy: 0.88700	Precision: 0.60657	Recall: 0.43400	F1: 0.50597	F2: 0.46018
# 	Total predictions: 15000	True positives:  868	False positives:  563	
#                                 False negatives: 1132	True negatives: 12437
# 
# 
# 4. test_classifier(clf_ada2, data_dict_rob, features_adah, folds=1000)
# clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
# features_adah=['poi','exercised_stock_options','other','expenses','total_money','incentives_ratio', 
#                 'from_this_person_to_poi',
#                'total_stock_value','payment_f', "incentives", 'payment_2']
# 
# AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
#                    n_estimators=50, random_state=None)
# 
# Accuracy: 0.88707	Precision: 0.61860	Recall: 0.39900	F1: 0.48511	F2: 0.42949
# 	Total predictions: 15000	True positives:  798	False positives:  492	
#                                 False negatives: 1202	True negatives: 12508
# 
# 
# 5. test_classifier(clf_ada2, data_dict_rob, features_adai, folds=1000)
# clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
# features_adai=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio',
#                'from_this_person_to_poi' ,
#                'incentives', 'payment_2',
#                'from_this_person_to_poi_ratio']
# 
# AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
#                    n_estimators=50, random_state=None)
# 
# Accuracy: 0.87847	Precision: 0.56629	Recall: 0.37800	F1: 0.45337	F2: 0.40493
# 	Total predictions: 15000	True positives:  756	False positives:  579	
#                                 False negatives: 1244	True negatives: 12421
# 
# 
# 
# 
# 6. test_classifier(clf_ada2_rf2, data_dict_rob, features_rfd, folds=1000)
# #rf2 + ada2 + features_rfd
# clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42)
# clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
# clf_ada2_rf2=AdaBoostClassifier(base_estimator=clf_rf2, n_estimators=500, learning_rate=1.5, random_state=42)
# 
# features_rfd=['poi', 'bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
#               'total_payments','incentives_ratio'  ]
# 
# AdaBoostClassifier(algorithm='SAMME.R',
#                    base_estimator=RandomForestClassifier(bootstrap=True,
#                                                          class_weight=None,
#                                                          criterion='gini',
#                                                          max_depth=7,
#                                                          max_features=2,
#                                                          max_leaf_nodes=None,
#                                                          min_impurity_decrease=0.0,
#                                                          min_impurity_split=None,
#                                                          min_samples_leaf=1,
#                                                          min_samples_split=2,
#                                                          min_weight_fraction_leaf=0.0,
#                                                          n_estimators=200,
#                                                          n_jobs=None,
#                                                          oob_score=False,
#                                                          random_state=42,
#                                                          verbose=0,
#                                                          warm_start=False),
#                    learning_rate=1.5, n_estimators=500, random_state=42)
# 
# Accuracy: 0.88053	Precision: 0.60317	Recall: 0.30400	F1: 0.40426	F2: 0.33748
# 	Total predictions: 15000	True positives:  608	False positives:  400	
#                                 False negatives: 1392	True negatives: 12600

# In[15]:


def pass_clf_short(algo_list, features_list,output_list,data_df,clf):
    
    # features_list includes poi 
    # algo_list is feature list without poi
    # output_list includes poi and poi_pred 
    #train and test set
    set_train, set_test = train_test_split(data_df, test_size = 0.3 #, random_state=42
                                          )
    features_train = set_train[algo_list]
    labels_train=set_train.poi
   
    features_test = set_test[algo_list ]
    labels_test=set_test.poi
          
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
    
    #fit and predict the algo
    fit = clf.fit(features_train, labels_train)
    dt_pred = clf.predict(features_test)
    labels_pred=clf.predict(features_test)
    
    #score of algo  
    dt_score = clf.score(features_test, labels_test)
    dt_precision = precision_score(labels_test, dt_pred)
    dt_recall = recall_score(labels_test, dt_pred)
    importance = clf.feature_importances_
    rf_pred = clf.predict(features_test)

    # creation of dataframe with prediction poi/nonpoi
    # 7. output list with poi and poi_pred
   
    df_pred_final = features_test.copy()
    df_pred_final ['poi'] = data_df['poi']
    df_pred_final['poi_pred'] = rf_pred
    df_pred_final= df_pred_final.loc[ :,(output_list)] 
    
    # print of information
   # print('data frame shape',data_df.shape)
   # print('train frame shape',features_train.shape, labels_train.shape)
   # print('test frame shape',features_test.shape, labels_test.shape)
   # print("\n")
    
    print("\n","CONFUSION MATRIX","\n",confusion_matrix(labels_test,labels_pred),"\n")
   # print("\n")

    print("CLASSIFICATION REPORT","\n",classification_report(labels_test, labels_pred))
    print("\n")
    
   # print("PRECISSION","\n",dt_precision)
   # print("\n")
    
   # print("RECALL","\n",dt_recall)
   # print("\n")
    
    # print Features importance
    
   # fi_df=pd.DataFrame({'feature':list(algo_list),'importance':clf.feature_importances_}
    #              ).sort_values('importance',ascending=False )
   # print(fi_df)
    
    
    # print dataframe with poi and poi prediction   
    data_poi = df_pred_final[df_pred_final['poi'] ==True]
    
    pred_poi = df_pred_final[df_pred_final['poi_pred'] ==True]
      
    result_poi = data_poi.append(pred_poi)
    

    return result_poi   


# In[24]:


# 1. algorithm and features to test

clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42 )
features_rfc=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio' ]

# clf to change
clf = clf_rf2

#features_list to change
algo_list=features_rfc

# features list for function
poi=['poi']
algo_list= list(set(algo_list).difference(set(poi)))

features_list=algo_list.copy()
features_list.insert(0,'poi')

output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# pass algorithm 

pass_clf_short(algo_list, features_list,output_list,data_df,clf)


# In[28]:


# 2.algorithm and features to test

clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42)
features_rfd=['poi', 'bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
              'total_payments','incentives_ratio'  ]


# clf to change
clf = clf_rf2

#features_list to change
algo_list=features_rfd

# features list for function
poi=['poi']
algo_list= list(set(algo_list).difference(set(poi)))

features_list=algo_list.copy()
features_list.insert(0,'poi')

output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# pass algorithm 

pass_clf_short(algo_list, features_list,output_list,data_df,clf)


# In[36]:


# 3. algorithm and features to test

clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
features_adag=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio',
               'from_this_person_to_poi' ]


# clf to change
clf = clf_ada2

#features_list to change
algo_list=features_adag

# features list for function
poi=['poi']
algo_list= list(set(algo_list).difference(set(poi)))

features_list=algo_list.copy()
features_list.insert(0,'poi')

output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# pass algorithm 

pass_clf_short(algo_list, features_list,output_list,data_df,clf)


# In[64]:


# 4. algorithm and features to test

clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
features_adah=['poi','exercised_stock_options','other','expenses','total_money','incentives_ratio', 
                'from_this_person_to_poi',
               'total_stock_value','payment_f', "incentives", 'payment_2']

# clf to change
clf = clf_ada2

#features_list to change
algo_list=features_adah

# features list for function
poi=['poi']
algo_list= list(set(algo_list).difference(set(poi)))

features_list=algo_list.copy()
features_list.insert(0,'poi')

output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# pass algorithm 

pass_clf_short(algo_list, features_list,output_list,data_df,clf)


# In[72]:


# 5. algorithm and features to test

clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
features_adai=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio',
               'from_this_person_to_poi' ,
               'incentives', 'payment_2',
               'from_this_person_to_poi_ratio']

# clf to change
clf = clf_ada2

#features_list to change
algo_list=features_adai

# features list for function
poi=['poi']
algo_list= list(set(algo_list).difference(set(poi)))

features_list=algo_list.copy()
features_list.insert(0,'poi')

output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# pass algorithm 

pass_clf_short(algo_list, features_list,output_list,data_df,clf)


# In[85]:


# 6. algorithm and features to test

clf_ada2_rf2=AdaBoostClassifier(base_estimator=clf_rf2, n_estimators=500, learning_rate=1.5, random_state=42)

features_rfd=['poi', 'bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
              'total_payments','incentives_ratio'  ]

# clf to change
clf = clf_ada2_rf2

#features_list to change
algo_list=features_rfd

# features list for function
poi=['poi']
algo_list= list(set(algo_list).difference(set(poi)))

features_list=algo_list.copy()
features_list.insert(0,'poi')

output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# pass algorithm 

pass_clf_short(algo_list, features_list,output_list,data_df,clf)

