from __future__ import division
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy.linalg as linalg
import pandas as pd

#load the data
dataTrain= np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

# split the data into input variables and labels
XTrain = dataTrain[:,:-1]
YTrain = dataTrain[:,-1]

#visualisation of the data
print(XTrain.shape)
#sc = StandardScaler()
#data_scaled = sc.fit_transform(XTrain)
data = dataTrain[:,:-1] 
print(data.shape)
mean = np.mean(data, axis=0)
#center data
data_scaled= data-mean

#covariance matrix with scaled data
dataset_mat = np.cov(data_scaled.T)   
eigenValues, eigenVectors = linalg.eig(dataset_mat)
idx = eigenValues.argsort()[::-1]  
#sorted eigenvalues and eigenvectors 
e_Values = eigenValues[idx]
e_Vectors = eigenVectors[:,idx]

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [((e_Values[i]), e_Vectors[:,i]) for i in range(len(e_Values))]


# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
#projection matrix
matrix_w = np.hstack((eig_pairs[0][1].reshape(13,1), 
                      eig_pairs[1][1].reshape(13,1)
                    ))






#kmeans clustering
startingpoint = np.vstack((XTrain[0,],XTrain[1,]))

kmeans_model = KMeans(algorithm='full', copy_x=True, init=startingpoint,max_iter=300,\
                      n_clusters=2, n_init=1).fit(data_scaled)

#centroid values the algorithm generated

labels = kmeans_model.labels_
y_predict=kmeans_model.fit_predict(data_scaled)
#print(y_predict)

centroids = kmeans_model.cluster_centers_
print("The Centroids:",centroids)




#Projecting the centered data
transformed = np.dot(data_scaled,matrix_w) 
print(transformed) 


plt.scatter(transformed[:,0], transformed[:,1],c=y_predict)    
 
plt.title('Plot of projected data pointsfor 2 PCs')











#Projecting the centroids 
transformed = np.dot(centroids,matrix_w) 
print(transformed) 
plt.scatter(transformed[:,0], transformed[:,1],color='red')    
#plt.scatter(data_scaled[:,0], data_scaled[:,1], c=transformed)  


plt.title('Plot of projected centroids ')
plt.show()


