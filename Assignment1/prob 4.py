import numpy as np
import matplotlib.pyplot as plt
import scipy


data = np.loadtxt('smoking.txt')
type(data)

#extract FEV1 and age values from data
FEV1 = data[:,1]
age_group = data[:,0]

#plotting
plt.bar(age_group,FEV1) 
plt.ylabel('FEV1 values')
plt.xlabel('age')
plt.title('age vs FEV1')
plt.show() 

#pearson correlation
pearson_correlation=scipy.stats.pearsonr(age_group,FEV1)
print("pearson_correlation with p-value", pearson_correlation)




















