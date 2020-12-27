#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
 

# Import needed along the file
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import datasets
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, precision_score, recall_score 
from sklearn.metrics import classification_report, f1_score, make_scorer

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier



# Define Function to correct data, remove outlier and define new features
### Task 1: Select what features to use.
### The first feature must be "poi".
### Load the dictionary containing the dataset
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.



# Function to do taks 1, 2 and 3
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
    
##### Loading the dictionary containing the dataset

    data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
   
    
##### Right data to be corrected in the dataset

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
    
    
##### Entries to be deleted in the dataset
    list_out=['TOTAL','THE TRAVEL AGENCY IN THE PARK']
    
##### Deleting the entries in the dictionary. 
    for name in list_out:
        data_dict.pop(name,0)
        
        
##### New features

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
        

        v["incentives_ratio"] =  (float(  incentives) / float( total_money) if
                          incentives not in [0, "NaN"] and total_money
                           not in [0, "NaN"] else 0.0)
                
        incentives_ratio= v["incentives_ratio"] 
        
        
        v["from_this_person_to_poi_ratio"] =  (float( from_this_person_to_poi) / float( shared_receipt_with_poi) if
                         from_this_person_to_poi not in [0, "NaN"] and shared_receipt_with_poi
                           not in [0, "NaN"] else 0.0)
                
        from_this_person_to_poi_ratio= v["from_this_person_to_poi_ratio"] 
              
             
    features_original_list.append("incentives") 
    features_original_list.append("incentives_ratio")
    features_original_list.append("total_money")                    
    features_original_list.append("from_this_person_to_poi_ratio")     
    
    

    my_dataset = data_dict

##### data frame
    data_df = pd.DataFrame.from_dict(data_dict, orient='index')
    data_df= data_df.loc[ :,(features_amendm_list )]  

    data_df= data_df.replace('NaN', 0.0)
    data_df=round(data_df,2)

    #print('data frame shape',data_df.shape)
    return data_df



# Run Function to correct data, remove outlier and define new features
##### 1. original features
features_list0=[ 'poi',
                               'salary', 'bonus', 'long_term_incentive',
                               'deferred_income','deferral_payments', 'loan_advances',
                               'other', 'expenses','director_fees',  'total_payments',
                               'exercised_stock_options', 'restricted_stock',
                               'restricted_stock_deferred', 'total_stock_value',
                               'from_messages', 'to_messages', 
                               'from_poi_to_this_person', 'from_this_person_to_poi','shared_receipt_with_poi']

##### 2. features including new ones and the ones used for new features creation and for algorithm 
features_list1=[ 'poi',
                 'salary', 'bonus', 'long_term_incentive',
                 'deferred_income','deferral_payments',
                 'other', 'expenses','total_payments',
                 'exercised_stock_options', 'restricted_stock', 'total_stock_value',
                 'from_messages', 'to_messages', 
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
               
               "incentives_ratio", "from_this_person_to_poi_ratio",
               "incentives","total_money"   , 'payment_2', 'payment_f'
             ]

##### 3. run the function 
data_df=create_df (features_list0,features_list1)




# Features creation with PCA

from sklearn.decomposition import PCA
        
payment_f=['salary', 'bonus', 'long_term_incentive', 'other', 'expenses']
payment_2=['salary', 'other', 'expenses']

pca = PCA(n_components=1)
pca.fit(data_df[payment_2])
pcaComponents = pca.fit_transform(data_df[payment_2])

data_df['payment_2']=pcaComponents

pca = PCA(n_components=1)
pca.fit(data_df[payment_f])
pcaComponents = pca.fit_transform(data_df[payment_f])

data_df['payment_f']=pcaComponents

# create my_dataset with the features to be used
data_df  = data_df .loc[ :,( features_list1)]

data_df=round(data_df,2)

features_list=[ 'poi','bonus','other', 'expenses','total_payments',
                 'exercised_stock_options', 'total_stock_value',
               'from_this_person_to_poi',
              "incentives_ratio", "from_this_person_to_poi_ratio",
               "incentives","total_money" , 'payment_f', 'payment_2'  ]

data_df  = data_df .loc[ :,( features_list)]
data_dict=data_df.to_dict('index')
my_dataset=data_dict

### Task 4: Try a varity of classifiers
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Definiton of different classifier to test     
# 0. RandomForest and Ada Boost basic clasifier
clf0_1=RandomForestClassifier()
clf0_2=AdaBoostClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. 
#1. RandomForest finetune with some features
clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42)
features_rfd=['poi', 'bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
              'total_payments','incentives_ratio'  ]

# 2. Ada Boost finetune with some features
clf_ada2=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.5, random_state=42)
features_adah=['poi','exercised_stock_options','other','expenses','total_money','incentives_ratio', 
                'from_this_person_to_poi',
               'total_stock_value','payment_f', "incentives", 'payment_2']


# 3. Ada Boost finetune with RadomForest base estimator with some features
clf_rf2=RandomForestClassifier(n_estimators=200, max_depth=7, max_features = 2, random_state=42)
clf_ada2_rf2=AdaBoostClassifier(base_estimator=clf_rf2, n_estimators=500, learning_rate=1.5, random_state=42)
features_rfd=['poi', 'bonus','exercised_stock_options', 'total_stock_value', 'expenses' ,
              'total_payments','incentives_ratio'  ]


PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


# Define tester: this is the one provided in tester.py
from sklearn.model_selection import StratifiedShuffleSplit
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
        print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, 
                                           false_negatives, true_negatives))
        print("")

               
    except:
        print("Got a divide by zero when trying out:", clf)

# Define pipeline for scaler and classifier
from sklearn.preprocessing import RobustScaler
clf_trial=clf0_1
pipe_clf= Pipeline([['rob_sc',RobustScaler() ],['clf', clf_trial]])
clf_0_1p=pipe_clf   
# Test the classifier
print('RandomForestClassifier()')
test_classifier(clf_0_1p, my_dataset, features_list, folds=1000)

# Define pipeline for scaler and classifier
clf_trial=clf0_2
pipe_clf= Pipeline([['rob_sc',RobustScaler() ],['clf', clf_trial]])
clf_0_2p=pipe_clf
# Test the classifier
print('AdaBoostClassifier()')
test_classifier(clf_0_2p, my_dataset, features_list, folds=1000)

# Define pipeline for scaler and classifier
clf_trial=clf_rf2
pipe_clf= Pipeline([['rob_sc',RobustScaler() ],['clf', clf_trial]])
clf_rf2p=pipe_clf
# Test the classifier
print('RandomForestClassifier(finetune)')
test_classifier(clf_rf2p, my_dataset, features_rfd, folds=1000)

# Define pipeline for scaler and classifier
clf_trial=clf_ada2
pipe_clf= Pipeline([['rob_sc',RobustScaler() ],['clf', clf_trial]])
clf_ada2p=pipe_clf
# Test the classifier
print('AdaBoostClassifier(finetune)')
test_classifier(clf_ada2p, my_dataset, features_adah, folds=1000)

# Define pipeline for scaler and classifier
clf_trial=clf_ada2_rf2
pipe_clf= Pipeline([['rob_sc',RobustScaler() ],['clf', clf_trial]])
clf_ada2_rf2p=pipe_clf
# Test the classifier
print('AdaBoostClassifier(base estimator RandonForest and finetune)')
test_classifier(clf_ada2_rf2p, my_dataset, features_rfd, folds=1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "wb") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "wb") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "rb") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "rb") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "rb") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list




clf=clf_ada2
my_dataset=data_dict
features_list=features_adah

dump_classifier_and_data(clf, my_dataset, features_list)

