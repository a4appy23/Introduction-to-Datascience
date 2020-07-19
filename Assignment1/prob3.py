
#hypothesis testing
from __future__ import division
import numpy as np
from scipy.stats import t
from tabulate import tabulate

 #load the data
data = np.loadtxt('smoking.txt') 
FEV1 = data[:,1]
smoker_column = data[:, 4] 
FEV = np.array(FEV1)
smoker_c = np.array(smoker_column)

#extract fev values from smoker column with values 0 and 1
FEV_non_smoker= FEV[smoker_c ==0]  #if value of smoker column is 0 
FEV_smoker = FEV[smoker_c ==1]
x1 = np.mean(FEV_non_smoker)
x2 = np.mean(FEV_smoker)
print("mean--", x1, x2)

#difference between the mean
mean_diff = x1-x2
print("mean_diff", mean_diff)
var1 = np.var(FEV_non_smoker)
var2 = np.var(FEV_smoker)
N1 = len(FEV_non_smoker)
N2 = len(FEV_smoker)

s1_sqr = np.std(FEV_smoker)**2
s2_sqr = np.std(FEV_non_smoker)**2

# Test statistic
standard_error = np.sqrt(var1/N1+var2/N2)
test_stats = mean_diff/standard_error
print("test statistics", test_stats)

#degree of freedom
DF = (var1/N1 + var2/N2)**2 / ((var1 / N1)**2 / (N1 - 1)  +  (var2 / N2)**2 / (N2 - 1) )
p = 2*t.cdf(test_stats, DF)
print("DF and p", DF, p)
print tabulate([['DF', DF], ['p', p],['test statistics', test_stats]], headers=['parameters', 'values'])

#q = ttest_ind(FEV_smoker, FEV_non_smoker, equal_var=False)
#print(q)








