import matplotlib.pyplot as plt
import numpy as np
#load the data
data = np.loadtxt('smoking.txt') 
t = data.shape
FEV1 = data[:,1]
smoker_column = data[:, 4] 
FEV = np.array(FEV1)
smoker_c = np.array(smoker_column)

#extract fev values from smoker column with values 0 and 1
non_smoker= FEV[smoker_c ==0]  #if value of smoker column is 0 
smoker = FEV[smoker_c ==1] #if value of smoker column is 1
print("The FEV of nonsmokers", non_smoker)
print("The FEV of smokers", smoker)

#calculate the mean of  FEV vaules of smoker and non smoker
fev_nonsmoker = np.mean(non_smoker)
fev_smoker = np.mean(smoker)
print("the mean of FEV_mean of non-smoker", fev_nonsmoker)
print("the mean of FEV_mean of smoker", fev_smoker)

#plotting using boxplot
fig,ax = plt.subplots()
bp1 = ax.boxplot([non_smoker,smoker], labels=['N','S'], patch_artist=True, positions=[1,2], boxprops=dict(facecolor="C0") )
ax.set_title('FEV1 mean values of nonsmokers and smokers')
ax.set_ylabel('FEV1 values')
ax.set_xlabel('N:Non-smokers and S:smokers')
plt.show()
