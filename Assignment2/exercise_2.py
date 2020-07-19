
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


#load the data
Train= np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
Test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

XTrain = Train[:,:-1]
YTrain = Train[:,-1]

XTest = Test[:,:-1]
YTest = Test[:,-1]

k_values = [1,3,5,7,9,11]
scores_1 = []
scores_3 = []
scores_5 = []
scores_7 = []
scores_9 = []
scores_11 = []
#KFold crossvalidation with splits=5
cv = KFold(n_splits=5)


def get_score(model,XTrainCV,XTestCV,YTrainCV,YTestCV):
    model.fit(XTrainCV,YTrainCV)
    return model.score(XTestCV,YTestCV)

#generating scores for different values of k from 1,3,5,7,9,11.

for train, test in cv.split(XTrain):
    XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train],XTrain[test],\
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

print("--k_values--",k_values)
print("--crossvalscores--:",np.mean(scores_1),np.mean(scores_3),np.mean(scores_5),np.mean(scores_7),np.mean(scores_9),np.mean(scores_11))

  
 #misclassification error   
MSE_1 = [1 - x for x in scores_1] 
MSE_3 = [1 - x for x in scores_3] 
MSE_5 = [1 - x for x in scores_5] 
MSE_7 = [1 - x for x in scores_7] 
MSE_9 = [1 - x for x in scores_9] 
MSE_11 = [1 - x for x in scores_11] 
print("--classification error---")
print( np.mean(MSE_1),np.mean(MSE_3),np.mean(MSE_5),np.mean(MSE_7),\
      np.mean(MSE_9),np.mean(MSE_11))


#determining kbest by sorting in ascending order
K_best = min(np.mean(MSE_1),np.mean(MSE_3),np.mean(MSE_5),np.mean(MSE_7),\
      np.mean(MSE_9),np.mean(MSE_11))
print("--minimum classification error for K_best---")
print(K_best)
print("best k-value:3")

 
    

                                   



    
    
    
    
   
  
    
    