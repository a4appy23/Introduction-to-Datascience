# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the same order as their associated eigenvalues

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA


data = np.loadtxt('pca_toydata.txt')
X_std = StandardScaler().fit_transform(data)
#center the data
#center data
#mean = np.mean(data, axis=0)
mean = np.mean(X_std, axis=0)
data_scaled= data-mean

dataset_mat = np.cov(X_std.T)   
eigenValues, eigenVectors = linalg.eig(dataset_mat)
idx = eigenValues.argsort()[::-1]   
e_Values = eigenValues[idx]
e_Vectors = eigenVectors[:,idx]



#variance vs pc index
var = (e_Values/sum(e_Values))
var_percent = var*100

labels = ['PC'+ str(x) for x in range(1, len(var)+1)]
plt.bar(x=range(1, len(var)+1), height = var,\
        tick_label=labels)
#that first principal component is responsible for 93.24% variance
#that second principal component is responsible for 6.7% variance
plt.title('bar-plot of variance vs PC index')
plt.xlabel('number of components')
plt.ylabel('Projected variance')
plt.show()
print("The variance:",var)



