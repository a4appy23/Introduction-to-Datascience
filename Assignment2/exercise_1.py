
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
#load the data
Train= np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
Test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
# split the training and test data in to input variables and labels
XTrain = Train[:,:-1]
YTrain = Train[:,-1]
XTest = Test[:,:-1]
YTest = Test[:,-1]
#print(len(YTrain))
#print(len(Train),len(YTrain))

#Apply the knn classifier to the model
knn = KNeighborsClassifier(n_neighbors=1)
#fitting the model
knn.fit(XTrain,YTrain)

# predict the model
predict = knn.predict(XTest)
print("----------predictions from the classifier-------")
print(predict)
#accuracy score 
accuracy = accuracy_score(YTest,predict)
print('The accuracy_score of our classifier is ', accuracy)
print("accuracy percent",accuracy*100)
scores = cross_val_score(knn, XTrain, YTrain, cv=5) 
crossval_mean =np.mean(scores)*100
print("crossvalidation accuracy score",crossval_mean) 