#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import numpy as np

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(features_train, labels_train)
predictions = model.predict(features_test)
accuracy = accuracy_score(labels_test, predictions)
print(accuracy)
print(len(predictions[predictions == 1]))
print(len(features_test))
labels_test = np.array(labels_test)
print(len(labels_test[labels_test == 0]))
print(len(predictions[np.all([predictions == 1, predictions == labels_test])]))
print(precision_score(labels_test, predictions))
print(recall_score(labels_test, predictions))
