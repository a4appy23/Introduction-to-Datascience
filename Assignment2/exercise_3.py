
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#load the data
dataTrain= np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
# split the data into input variables and labels
XTrain = dataTrain[:,:-1]
YTrain = dataTrain[:,-1]

XTest = dataTest[:,:-1]
YTest = dataTest[:,-1]


#Assign the knn classifier with k value = 3
knn_best = KNeighborsClassifier(n_neighbors=3)
#fitting the model
knn_best.fit(XTrain,YTrain)
# predict the model
predict = knn_best.predict(XTest)
#accuracy score 
accuracy = accuracy_score(XTest,predict)
accuracy_t = accuracy_score(YTest,predict)

print('The accuracy of our classifier is ', accuracy)
print("The accuracy percentage ",accuracy*100)

print('The accuracy of our classifier is ', accuracy_t)
print("The accuracy percentage ",accuracy_t*100)


#KFold crossvalidation with splits=5
cv = KFold(n_splits=5)

for train, test in cv.split(XTrain):
    XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train],XTrain[test],\
                                           YTrain[train],YTrain[test]

#perform 5 fold crossvalidation and calculate the score
scores = cross_val_score(knn_best, XTrain, YTrain, cv=5) 
crossval_mean =np.mean(scores)*100

print("crossvalidation score",crossval_mean) 
#plotting 
plt.plot(scores)
plt.show()

















