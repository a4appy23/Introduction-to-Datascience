import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#load the data
dataTrain= np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
data = dataTrain[:,:-1] 
print(data.shape)
mean = np.mean(data, axis=0)
#center data
data_scaled= data-mean
 
#scaling the data
#sc = StandardScaler()  
#data_scaled = sc.fit_transform(data) 

#covariance matrix with scaled data
dataset_mat = np.cov(data_scaled.T)   
eigenValues, eigenVectors = linalg.eig(dataset_mat)
idx = eigenValues.argsort()[::-1]  
#sorted eigenvalues and eigenvectors 
e_Values = eigenValues[idx]
e_Vectors = eigenVectors[:,idx]

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [((e_Values[i]), e_Vectors[:,i]) for i in range(len(e_Values))]
print(dataset_mat.shape)

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
#projection matrix
matrix_w = np.hstack((eig_pairs[0][1].reshape(13,1), 
                      eig_pairs[1][1].reshape(13,1)
                    ))
print(matrix_w.shape)

#Projecting the centered data
transformed = np.dot(data_scaled,matrix_w) 
print(transformed) 
plt.scatter(transformed[:,0], transformed[:,1])    
 


plt.title('Plot of projected data pointsfor 2 PCs ')

