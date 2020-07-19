import numpy as np

data = np.loadtxt('smoking.txt') #load the data
print(data)
t = data.shape
print(t)
#extract the values from the column
age = data[:,0]
FEV1 = data[:,1]
height = data[:,2]
gender = data[:,3]
weight = data[:,5]
smoker_column = data[:, 4]

#creating array of FEV1 and smoker values 
FEV = np.array(FEV1)
smoker_c = np.array(smoker_column)

#extract FEV values with values 0 and 1(nonsmoker =0,smoker =1)
non_smoker= FEV[smoker_c ==0]  
smoker = FEV[smoker_c ==1] 

#calculate the mean of  FEV vaules of smoker and non smoker
fev_nonsmoker = np.mean(non_smoker)
fev_smoker = np.mean(smoker)
print("the mean of FEV_mean of non-smoker", fev_nonsmoker)
print("the mean of FEV_mean of non-smoker", fev_smoker)











