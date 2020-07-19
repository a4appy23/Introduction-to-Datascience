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
print("The Centroids:",centroids)
labels = kmeans_model.labels_
y_predict=kmeans_model.fit_predict(data_scaled)
#print(y_predict)

#visualising the model for different point in the dataset with the centroids
#for the different values of I and j in the plot  the data is clustered and the corresponding centroid is printed
a=[0,1,2,3,4,5,6,7,8,9,10,11]
for i in a:
    j=i+1
    plt.scatter(data_scaled[:,i],data_scaled[:,j], c=y_predict)
    print(i,j)

#plt.show()
    plt.scatter(centroids[:,i],centroids[:,j],marker = "o",color='red', s=50, linewidths = 3)
    print(centroids[:,i],centroids[:,j])
#plt.xlim(-1,6)
#plt.ylim(-1,6)
    plt.title('kmeans clustering with centroid for different values dataset')
    plt.show()
    i=i+1
    j=i+1
    

   




