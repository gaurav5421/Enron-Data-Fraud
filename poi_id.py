#!/usr/bin/python

from __future__ import division
import sys
import pickle
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import cross_validation
from time import time
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.feature_selection import SelectPercentile,f_classif

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'
				, 'salary'
				# , 'deferral_payments'
				, 'total_payments'
				, 'loan_advances'
				, 'bonus'
				# , 'restricted_stock_deferred'
				, 'deferred_income'
				, 'total_stock_value'
				, 'expenses'
				, 'exercised_stock_options'
				# , 'other'
				, 'long_term_incentive'
				, 'restricted_stock'
				# , 'director_fees'
				# , 'to_messages'
				# , 'from_poi_to_this_person'
			 #    , 'from_messages'
				# , 'from_this_person_to_poi'
				, 'shared_receipt_with_poi'
				] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK') # Removed this since this data point does not make any sense in the analysis we are trying to do.

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
# for k,v in data_dict.items():
# 	if data_dict[k]['salary']  == 'NaN' or data_dict[k]['bonus'] == 'NaN':
# 		data_dict[k]['bonus_to_sal_ratio'] = 'NaN'
# 	else:
# 		data_dict[k]['bonus_to_sal_ratio'] = data_dict[k]['bonus']/data_dict[k]['salary']
# 	#print (data_dict[k]['sal_to_bonus_ratio'])
# features_list.append("bonus_to_sal_ratio")



my_dataset = data_dict
# print "Total number of data points:" , len(my_dataset)
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels,test_size=0.30, random_state=42)

#Feature Selection
selector = SelectKBest(f_classif, k = 10)
selected_features = selector.fit_transform(features, labels)
scores = selector.scores_
print (scores)

### Task 4: Try a varity of classifiers
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

print "Prediction score for Naive Bayes classifier:", (accuracy_score(pred, labels_test))

print(confusion_matrix(labels_test,pred))
print('\n')
print(classification_report(labels_test,pred))

pl = [('SelectKBest', SelectKBest()), ('Dtree', tree.DecisionTreeClassifier(random_state=42))]
pipeline = Pipeline(pl)
best_cv=cross_validation.StratifiedShuffleSplit(labels,100,random_state=42)
param_grid = {'SelectKBest__k':range(1,4), 'Dtree__criterion':["entropy","gini"],'Dtree__min_samples_split':[2,5,10,20,40]}
grid = GridSearchCV(pipeline,param_grid,scoring='f1',cv=best_cv,verbose = 3)
grid.fit(selected_features,labels) 
clf = grid.best_estimator_
pred = clf.predict(features_test)
print "Prediction score for Decision Tree classifier:", (accuracy_score(pred, labels_test))

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
