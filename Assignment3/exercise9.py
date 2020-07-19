from __future__ import division
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#load the data
dataTrain= np.loadtxt('MNIST_179_digits')
datalabel= np.loadtxt('MNIST_179_labels')



sc = StandardScaler()
data_scaled = sc.fit_transform(dataTrain)

startingpoint = np.vstack((dataTrain[0,],dataTrain[1,]))

kmeans_model = KMeans(algorithm='full', copy_x=True, init=startingpoint,max_iter=300,\
                      n_clusters=3, n_init=1).fit(data_scaled)