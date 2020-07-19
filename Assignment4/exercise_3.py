import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#load the data
data= np.loadtxt('pca_toydata.txt')
print(data.shape)

#mean = np.mean(data, axis=0)
#data_scaled = (data - mean)
#scaling the data
sc = StandardScaler()  
data_scaled = sc.fit_transform(data) 

#covariance matrix with scaled data
dataset_mat = np.cov(data_scaled.T)   
eigenValues, eigenVectors = linalg.eig(dataset_mat)
idx = eigenValues.argsort()[::-1]  
#sorted eigenvalues and eigenvectors 
e_Values = eigenValues[idx]
e_Vectors = eigenVectors[:,idx]
evals_r = np.abs(np.real(e_Values))
evecs_r = np.real(e_Vectors) #eigenvectors should all have unit length 1
#projection matrix

print(dataset_mat.shape)
#projection matrix to reduce d dimensional space to 2D subspace by choosing the first 2 PCs
W = np.hstack((evecs_r[:,0].reshape(2,1), evecs_r[:,1].reshape(2,1)))
#Projecting the centered data
transformed = np.dot(data_scaled,W) 
print(transformed) 
plt.scatter(transformed[:,0], transformed[:,1])    
#plt.scatter(data_scaled[:,0], data_scaled[:,1], c=transformed)  

plt.title('Plot of projected data pointsfor 2 PCs ')
plt.show()

plt.title('Plot of projected data after  ')
#removing the last row for last two points
transformed_n=transformed[:-2:]
print(transformed_n)
plt.scatter(transformed_n[:,0], transformed_n[:,1]) 





