from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#load data
dataTrain= np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
XTrain = dataTrain[:,:-1]
YTrain = dataTrain[:,-1]
XTest = dataTest[:,:-1]
YTest = dataTest[:,-1]
#plotting the data before preprocessing for visualization
plt.hist(XTrain)
plt.xlabel('XTrain values')
plt.ylabel('Count')
plt.show()

plt.hist(XTest)
plt.xlabel('XTest values')
plt.ylabel('Count')
plt.show()

#Applying preprocessing techniques
#method 1

scaler=preprocessing.StandardScaler( ).fit(XTrain)
XTrainN=scaler.transform(XTrain)
XTestN = scaler.transform(XTest)
print("standard deviation",np.std(XTrainN))
print("-The mean -",np.mean(XTrainN),np.mean(XTestN))

#plotting the training and test data
plt.hist(XTrainN)
plt.xlabel('XTrain preprocessed values')
plt.ylabel('Count')
plt.legend()
plt.show()

plt.hist(XTestN)
plt.xlabel('XTest preprocessed values')
plt.ylabel('Count')
plt.show()

#Applying preprocessing techniques
#method 2
scaler=preprocessing.StandardScaler( ).fit(XTrain)
XTrainN2=scaler.transform(XTrain)
scaler=preprocessing.StandardScaler( ).fit(XTest)
XTestN2 = scaler.transform(XTest)
print("standard deviation 2",np.std(XTrainN2))
print("-mean-",np.mean(XTrainN2),np.mean(XTestN2))
#Applying preprocessing techniques
#method 3
XTotal = np.concatenate((XTrain,XTest),axis=0)
scaler=preprocessing.StandardScaler( ).fit(XTotal)
XTrainN3=scaler.transform(XTrain)
XTestN3 = scaler.transform(XTest)
print("standard deviation 3",np.std(XTrainN3))
print("--",np.mean(XTrainN3),np.mean(XTestN3))

#Applying the preprocessing from method 1
k_values = [1,3,5,7,9,11]

for i in k_values:       
    # Fitting k-NN on our scaled data set
    knn = KNeighborsClassifier(n_neighbors=i)
    #fitting the model
    knn.fit(XTrainN,YTrain)
    # predict the model
    predict = knn.predict(XTestN)
    #accuracy score 
    accuracy = accuracy_score(YTest,predict)  
    
    
    if i == 1:
        print("------the accuracy score of normalised knn ----")
        print("kval,----accuracy----")
        print(i, accuracy)        
    elif i == 3:
         print(i,accuracy)
    elif i == 5:
         print(i,accuracy)
    elif i == 7:
         print(i,accuracy)
    elif i == 9:
         print(i,accuracy)
    elif i == 11:
         print(i,accuracy)
                              
               
#KFold crossvalidation with splits=5
cv = KFold(n_splits=5)
scores_1 = []
scores_3 = []
scores_5 = []
scores_7 = []
scores_9 = []
scores_11 = []

#defing a getscore function to fit and score model
def get_score(model,XTrainN,XTestN,YTrainCV,YTestCV):
    model.fit(XTrainN,YTrainCV)
    return model.score(XTestN,YTestCV)

#crossvalidation scores for diiferent values of knn=1,3,5,7,9,11
for train, test in cv.split(XTrainN):
    XTrainCV, XTestCV, YTrainCV, YTestCV = XTrainN[train],XTrainN[test],\
                                           YTrain[train],YTrain[test]
                                        
    knn_1 = KNeighborsClassifier(n_neighbors=1)
    scores_1.append(get_score(knn_1,XTrainCV,XTestCV,YTrainCV,YTestCV)) 
    knn_3 = KNeighborsClassifier(n_neighbors=3)
    scores_3.append(get_score(knn_3,XTrainCV,XTestCV,YTrainCV,YTestCV)) 
                                   
    knn_5 = KNeighborsClassifier(n_neighbors=5)
    scores_5.append(get_score(knn_5,XTrainCV,XTestCV,YTrainCV,YTestCV)) 
    knn_7 = KNeighborsClassifier(n_neighbors=7)
    scores_7.append(get_score(knn_7,XTrainCV,XTestCV,YTrainCV,YTestCV)) 
    knn_9 = KNeighborsClassifier(n_neighbors=9)
    scores_9.append(get_score(knn_9,XTrainCV,XTestCV,YTrainCV,YTestCV)) 
    knn_11 = KNeighborsClassifier(n_neighbors=11)
    scores_11.append(get_score(knn_11,XTrainCV,XTestCV,YTrainCV,YTestCV)) 
#mean scores of different values of KNN model
print("---crossvalscore---")
print("k = 1",np.mean(scores_1),"k=3",np.mean(scores_3),"k=5",np.mean(scores_5),\
      "k=7",np.mean(scores_7),"k=9",np.mean(scores_9),"k=11",np.mean(scores_11))
    
 #misclassification error   
MSE_1 = [1 - x for x in scores_1] 
MSE_3 = [1 - x for x in scores_3] 
MSE_5 = [1 - x for x in scores_5] 
MSE_7 = [1 - x for x in scores_7] 
MSE_9 = [1 - x for x in scores_9] 
MSE_11 = [1 - x for x in scores_11] 
print("--classificationerror---")
print(np.mean(MSE_1),np.mean(MSE_3),np.mean(MSE_5),np.mean(MSE_7),\
      np.mean(MSE_9),np.mean(MSE_11))


#determining kbest by selecting minimum 
K_best = min(np.mean(MSE_1),np.mean(MSE_3),np.mean(MSE_5),np.mean(MSE_7),\
      np.mean(MSE_9),np.mean(MSE_11))
print("--minimum classification error for kbest---", K_best)
print("The best kvalue : 3")

#evaluating the performance by choosing kbest with kvalue=3

knn = KNeighborsClassifier(n_neighbors=3)
#fitting the model
knn.fit(XTrainN,YTrain)
# predict the model
predict = knn.predict(XTestN)
#accuracy score 
accuracy = accuracy_score(YTest,predict)

print('The accuracy of our kbest with knn-3 is ', accuracy*100)








