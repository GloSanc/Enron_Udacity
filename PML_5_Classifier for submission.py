#!/usr/bin/env python
# coding: utf-8

# # 5. Testing classifier with tester.py tester function

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



import pickle
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

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
# 3. features list without poi . The one to pass to the algo
poi=['poi']
algo_list= list(set(features_new).difference(set(poi)))

# 4. features list with poi
features_list=algo_list.copy()
features_list.insert(0,'poi')
              

features_list_new=list(set(features_list1+features_new))





# 5. output list with poi and poi_pred
output_list=features_list.copy()
output_list.insert(1,'poi_pred')

# 6. orgianl features list without poi, with poi and poi_pred
algo_list0= list(set(features_list).difference(set(poi)))
algo_list0= list(set(algo_list0).difference(set(features_new)))
output_list0=features_list1.copy()
output_list0.insert(1,'poi_pred')

print (features_list)


# In[4]:


#!/usr/bin/python3

""" 
    A general tool for converting data from the
    dictionary format to an (n x k) python list that's 
    ready for training an sklearn algorithm

    n--no. of key-value pairs in dictonary
    k--no. of features being extracted

    dictionary keys are names of persons in dataset
    dictionary values are dictionaries, where each
        key-value pair in the dict is the name
        of a feature, and its value for that person

    In addition to converting a dictionary to a numpy 
    array, you may want to separate the labels from the
    features--this is what targetFeatureSplit is for

    so, if you want to have the poi label as the target,
    and the features you want to use are the person's
    salary and bonus, here's what you would do:

    feature_list = ["poi", "salary", "bonus"] 
    data_array = featureFormat( data_dictionary, feature_list )
    label, features = targetFeatureSplit(data_array)

    the line above (targetFeatureSplit) assumes that the
    label is the _first_ item in feature_list--very important
    that poi is listed first!
"""


import numpy as np


def featureFormat(
    dictionary,
    features,
    remove_NaN=True,
    remove_all_zeroes=True,
    remove_any_zeroes=False,
    sort_keys=False,
):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """

    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle

        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value == "NaN" and remove_NaN:
                value = 0
            tmp_list.append(float(value))

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == "poi":
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append(np.array(tmp_list))

    return np.array(return_list)


def targetFeatureSplit(data):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append(item[0])
        features.append(item[1:])

    return target, features


# In[5]:


def test_train(dataset, feature_list, folds=1000):
    data = featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
   # cv = StratifiedShuffleSplit(labels, folds, random_state=42)
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    #for train_idx, test_idx in cv:
    for train_idx, test_idx in cv.split (features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
    return features_train, features_test, labels_train, labels_test


# In[6]:


def test_classifier(clf, dataset, feature_list, folds=1000):
    data = featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
   # cv = StratifiedShuffleSplit(labels, folds, random_state=42)
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    #for train_idx, test_idx in cv:
    for train_idx, test_idx in cv.split (features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print(clf)
        print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
       # print("\n","CONFUSION MATRIX","\n",confusion_matrix(labels_test,labels_pred),"\n")
        print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print("")
               
    except:
        print("Got a divide by zero when trying out:", clf)


# In[8]:


# clf12. test_train(data_dict_rob, features_list, folds=1000)
clf= RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
test_classifier(clf, data_dict_rob, features_list, folds=1000)


# In[9]:


features_list2=['poi','total_money','expenses','incentives_ratio',
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
                            n_estimators=50, learning_rate=1.0, 
                          algorithm='SAMME.R', random_state=42)

test_classifier(clf_ada, data_dict_rob, features_list2, folds=1000)


# # 5.1. Comparing classifiers with different features
# 
# The objective then is try to find an algorithm that in combination with some features provides a precision and recall are both at least 0.3
# Lets see what are the result with different classfier and features

# In[7]:


# 4.3.1 rf1
# 4.3.1 rf2
# 4.3.1 rf3
# 4.3.2 rf4, rf5, rf6

# 4.4.1 ada1
# 4.4.1 ada2           
# 4.4.1 ada3
# 4.4.1 ada4


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
clf_ada6=AdaBoostClassifier(base_estimator=None, n_estimators=150, learning_rate=1.5, random_state=42)
clf_ada7=AdaBoostClassifier(base_estimator=None, n_estimators=150, learning_rate=2.0, 
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


# In[86]:


test_classifier(clf_rf1, data_dict_rob, features_rf1, folds=1000)


# In[65]:


test_classifier(clf_rf2, data_dict_rob, features_rf2, folds=1000)


# In[66]:


test_classifier(clf_rf3, data_dict_rob, features_rf3, folds=1000)


# In[67]:


test_classifier(clf_rf4, data_dict_rob, features_rf4, folds=1000)


# In[92]:


test_classifier(clf_rf4, data_dict_rob, features_rf1, folds=1000)


# In[68]:


test_classifier(clf_rf5, data_dict_rob, features_rf5, folds=1000)


# In[18]:


test_classifier(clf_rf5, data_dict_rob, features_rf1, folds=1000)


# In[22]:


test_classifier(clf_rf5, data_dict_rob, features_rf9, folds=1000)


# In[28]:


test_classifier(clf_rf5, data_dict_rob, features_ada1, folds=1000)


# In[25]:


test_classifier(clf_rf5, data_dict_rob, features_ada3, folds=1000)


# In[26]:


test_classifier(clf_rf5, data_dict_rob, features_ada4, folds=1000)


# In[27]:


test_classifier(clf_rf5, data_dict_rob, features_ada5, folds=1000)


# In[69]:


test_classifier(clf_rf6, data_dict_rob, features_rf6, folds=1000)


# In[70]:


test_classifier(clf_rf7, data_dict_rob, features_rf7, folds=1000)


# In[89]:


test_classifier(clf_rf8, data_dict_rob, features_rf8, folds=100)


# In[99]:


test_classifier(clf_rf9, data_dict_rob, features_rf9, folds=1000)


# In[23]:


test_classifier(clf_rf7, data_dict_rob, features_rf1, folds=1000)


# In[76]:


test_classifier(clf_ada1, data_dict_rob, features_ada1, folds=1000)


# In[77]:


test_classifier(clf_ada2, data_dict_rob, features_ada2, folds=1000)


# In[78]:


test_classifier(clf_ada3, data_dict_rob, features_ada3, folds=1000)


# In[79]:


test_classifier(clf_ada4, data_dict_rob, features_ada4, folds=1000)


# In[80]:


test_classifier(clf_ada5, data_dict_rob, features_ada5, folds=1000)


# In[93]:


test_classifier(clf_ada6, data_dict_rob, features_ada5, folds=100)


# In[ ]:


test_classifier(clf_ada7, data_dict_rob, features_ada5, folds=1000)


# In[24]:


test_classifier(clf_ada_rf1, data_dict_rob, features_ada_rf1, folds=100)


# In[83]:


test_classifier(clf_ada_rf2, data_dict_rob, features_ada_rf2, folds=1000)


# In[ ]:


## 5.2. Better classifier for submission


# In[8]:



features_rfa=['poi','bonus','exercised_stock_options', 'total_stock_value', 'other','expenses']
features_rfb=['poi','bonus','exercised_stock_options','other','from_this_person_to_poi','total_payments']
features_rfc=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio' ]
features_rfd=['poi', 'bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
              'total_payments','incentives_ratio'  ]
features_rfe=['poi','bonus','exercised_stock_options', 'total_stock_value', 'other','expenses',
               'total_payments', 'shared_receipt_with_poi','to_messages','from_this_person_to_poi']
features_rff=['poi','bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
              'total_payments', 'payment_tt', 'payment_f','incentives_ratio']
            
features_adag=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio',
               'from_this_person_to_poi' ]
features_adah=['poi','exercised_stock_options','other','expenses','total_money','incentives_ratio', 
                'from_this_person_to_poi',
               'total_stock_value','payment_f', "incentives", 'payment_2']
features_adai=['poi','exercised_stock_options','other', 'expenses','total_money','incentives_ratio',
               'from_this_person_to_poi' ,
               'incentives', 'payment_2',
               'from_this_person_to_poi_ratio']
                    

clf_rf1=RandomForestClassifier (random_state=42)

clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42)

clf_rf4=RandomForestClassifier(n_estimators=100, max_depth=None, max_features = 2,random_state=42)
clf_rf5=RandomForestClassifier (n_estimators=500, max_features='sqrt',max_depth=None,
                                min_samples_split=2, random_state=42)
clf_rf6=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators='warn',
                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
                        warm_start=False)

clf_ada1=AdaBoostClassifier()
clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
clf_ada3=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
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
clf_ada_rf3=AdaBoostClassifier((RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators='warn',
                        n_jobs=None, oob_score=False, random_state=42, verbose=0,
                        warm_start=False)),
                            n_estimators=50, learning_rate=1.0, 
                          algorithm='SAMME.R', random_state=42)


clf_ada_rf4=AdaBoostClassifier((RandomForestClassifier
                                     (n_estimators=200, max_depth=7, max_features = 2, random_state=42)),
                                    n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=42)


# In[33]:


test_classifier(clf_rf1, data_dict_rob, features_rfa, folds=1000)


# In[34]:


test_classifier(clf_rf1, data_dict_rob, features_rfb, folds=1000)


# In[35]:


test_classifier(clf_rf1, data_dict_rob, features_rfc, folds=1000)


# In[36]:


test_classifier(clf_rf1, data_dict_rob, features_rfd, folds=1000)


# In[37]:


test_classifier(clf_rf1, data_dict_rob, features_rfe, folds=1000)


# In[38]:


test_classifier(clf_rf1, data_dict_rob, features_rff, folds=1000)


# In[ ]:





# In[41]:


test_classifier(clf_rf2, data_dict_rob, features_rfa, folds=1000)


# In[42]:


test_classifier(clf_rf2, data_dict_rob, features_rfc, folds=1000)


# In[43]:


test_classifier(clf_rf2, data_dict_rob, features_rfd, folds=1000)


# In[46]:


test_classifier(clf_rf2, data_dict_rob, features_adag, folds=1000)


# In[47]:


test_classifier(clf_rf2, data_dict_rob, features_adah, folds=1000)


# In[48]:


test_classifier(clf_rf2, data_dict_rob, features_adai, folds=1000)


# In[50]:


test_classifier(clf_ada1, data_dict_rob, features_adag, folds=1000)


# In[51]:


test_classifier(clf_ada1, data_dict_rob, features_adah, folds=1000)


# In[52]:


test_classifier(clf_ada1, data_dict_rob, features_adai, folds=1000)


# In[81]:


test_classifier(clf_ada2, data_dict_rob, features_adag, folds=1000)


# In[55]:


test_classifier(clf_ada2, data_dict_rob, features_adah, folds=1000)


# In[53]:


test_classifier(clf_ada2, data_dict_rob, features_adai, folds=1000)


# In[56]:


test_classifier(clf_ada3, data_dict_rob, features_adag, folds=1000)


# In[57]:


test_classifier(clf_ada3, data_dict_rob, features_adah, folds=1000)


# In[58]:


test_classifier(clf_ada3, data_dict_rob, features_adai, folds=1000)


# In[70]:


test_classifier(clf_ada_rf4, data_dict_rob, features_rfc, folds=1000)


# In[78]:


test_classifier(clf_ada_rf4, data_dict_rob, features_rfd, folds=1000)


# In[65]:


test_classifier(clf_ada_rf4, data_dict_rob, features_adag, folds=1000)


# In[79]:


test_classifier(clf_ada_rf1, data_dict_rob, features_rfd, folds=10)


# In[87]:


test_classifier(clf_ada_rf2, data_dict_rob, features_rfd, folds=10)


# In[ ]:


test_classifier(clf_ada_rf3, data_dict_rob, features_rfd, folds=1000)


# In[9]:


#rf2 + ada2 + features_rfd
clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42)
clf_ada2_rf2=AdaBoostClassifier(base_estimator=clf_rf2, n_estimators=500, learning_rate=1.5, random_state=42)
features_rfd=['poi', 'bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
              'total_payments','incentives_ratio'  ]


# In[10]:


test_classifier(clf_ada2_rf2, data_dict_rob, features_rfc, folds=1000)


# In[12]:


test_classifier(clf_ada2_rf2, data_dict_rob, features_rfd, folds=1000)

