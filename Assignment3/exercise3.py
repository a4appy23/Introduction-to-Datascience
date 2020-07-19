from __future__ import division
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#load the data
dataTrain= np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')


# split the data into input variables and labels
XTrain = dataTrain[:,:-1]
YTrain = dataTrain[:,-1]

#visualisation of the data
print(XTrain.shape)

sc = StandardScaler()
data_scaled = sc.fit_transform(XTrain)
#plotting the data
plt.scatter(data_scaled[:,0], data_scaled[:,1])
plt.show()

startingpoint = np.vstack((XTrain[0,],XTrain[1,]))

kmeans_model = KMeans(algorithm='full', copy_x=True, init=startingpoint,max_iter=300,\
                      n_clusters=2, n_init=1).fit(data_scaled)

#centroid values the algorithm generated
centroids = kmeans_model.cluster_centers_
print(centroids)
labels = kmeans_model.labels_
y_predict=kmeans_model.fit_predict(data_scaled)
#print(y_predict)


#plt.subplot(1,2,2)
plt.scatter(data_scaled[:,0],data_scaled[:,1], c=y_predict)

#plt.show()
plt.scatter(centroids[:,0],centroids[:,1],marker = "o",color='red', s=50, linewidths = 3)
#plt.xlim(-1,6)
#plt.ylim(-1,6)
plt.show()

   




