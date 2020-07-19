# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the same order as their associated eigenvalues

import numpy as np
import numpy.linalg as linalg


data = np.loadtxt('murderdata2d.txt')
#center the data
#center data
mean = np.mean(data, axis=0)
data_scaled= data-mean


def pca(data):
    dataset_mat = np.cov(data_scaled)
    
    eigenValues, eigenVectors = linalg.eig(dataset_mat)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    return eigenValues,eigenVectors
print(pca(data))
  
 
    
    
    
    
    

