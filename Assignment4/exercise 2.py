from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

cell= np.loadtxt('diatoms.txt')

mean = np.mean(cell, axis=0)
#center data
data_scaled= cell-mean

#covariance matrix with scaled data
dataset_mat = np.cov(data_scaled.T)   
eigenValues, eigenVectors = linalg.eig(dataset_mat)
idx = eigenValues.argsort()[::-1]  
#sorted eigenvalues and eigenvectors 
e_Values = eigenValues[idx]
e_Vectors = eigenVectors[:,idx]
print(e_Values)

#variance vs pc index
var = (e_Values/sum(e_Values))
var_percent = var*100
plt.plot(var)
plt.title('plot of variance vs PC index')
plt.xlabel('number of components')
plt.ylabel('Projected variance')
plt.show()

c_var = np.cumsum(e_Values/sum(e_Values))


plt.plot(c_var)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
print(c_var)






