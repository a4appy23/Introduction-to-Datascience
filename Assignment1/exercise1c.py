import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#load the data
dataTrain= np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')


data = dataTrain[:,:-1]
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
print(e_Values)

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
plt.plot(var)
plt.title('plot of variance vs PC index')
plt.xlabel('number of components')
plt.ylabel('Projected variance')
plt.show()


#cumulative variance 
c_var = np.cumsum(e_Values/sum(e_Values))
print("cumulative variance:",c_var)
labels = ['PC'+ str(x) for x in range(1, len(c_var)+1)]
plt.bar(x=range(1, len(c_var)+1), height = c_var,\
        tick_label=labels)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('bar-plot of cumulative variance vs PC index')
plt.show()

plt.plot(c_var)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


#first 2 65 percent
#90 percent first 4 components and 95 6 component