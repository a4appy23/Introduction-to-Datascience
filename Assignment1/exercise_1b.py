from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
#load the data
data = np.loadtxt('murderdata2d.txt')
#scaling the data
sc = StandardScaler()  
data_scaled = sc.fit_transform(data) 

print(data_scaled)
mean =[0,0]
dataset_mat = np.cov(data_scaled.T)

eigenValues, eigenVectors = linalg.eig(dataset_mat)
idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print(eigenVectors)
# Compute the corresponding standard deviations
s0 = np.sqrt(eigenValues[0])
s1 = np.sqrt(eigenValues[1])
print(s0,s1)
#plotting data with projection
plt.scatter(data_scaled[:,0],data_scaled[:,1])
plt.plot([0, s0*eigenVectors[0,0]], [0, s0*eigenVectors[1,0]], 'r')
plt.plot([0, s1*eigenVectors[0,1]], [0, s1*eigenVectors[1,1]], 'r')
plt.title("the scaled data with the mean and standarddeviation")
#plt.axis('equal')















    