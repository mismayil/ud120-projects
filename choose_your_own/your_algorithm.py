#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary


########################## SVM #################################
### we handle the import statement and SVC creation for you here
# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", C=100)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
# model = clf.fit(features_train, labels_train)

#### store your predictions in a list named pred
# predictions = model.predict(features_test)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(predictions, labels_test)
# print(acc)
#
# try:
#     prettyPicture(model, features_test, labels_test)
# except NameError:
#     pass

####################### K NEAREST NEIGHBORS ############################
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
#
# K = 3
# model = KNeighborsClassifier(n_neighbors=K)
# model.fit(features_train, labels_train)
# predictions = model.predict(features_test)
# accuracy = accuracy_score(labels_test, predictions)
# print('K nearest neighbours accuracy = %f' % accuracy)

##################### RANDOM FOREST #########################
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# model = RandomForestClassifier(min_samples_split=50, n_estimators=1000)
# model.fit(features_train, labels_train)
# predictions = model.predict(features_test)
# accuracy = accuracy_score(labels_test, predictions)
# print('Random forest accuracy = %f' % accuracy)


################### ADABOOST ###############################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
model = AdaBoostClassifier(n_estimators=100)
model.fit(features_train, labels_train)
predictions = model.predict(features_test)
accuracy = accuracy_score(labels_test, predictions)
print('Adaboost accuracy = %f' % accuracy)
