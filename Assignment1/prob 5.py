import numpy as np
import matplotlib.pyplot as plt

#load the data
data = np.loadtxt('smoking.txt') #load the data
t = data.shape
FEV1 = data[:,1]
age_group = data[:,0]
smoker_column = data[:, 4] 
FEV = np.array(FEV1)
age = np.array(age_group)
smoker_c = np.array(smoker_column)
print(len(smoker_c))

#extract Fev values from smoker_c
FEV_non_smoker= FEV[smoker_c ==0]  #if value of smoker column is 0 
FEV_smoker = FEV[smoker_c ==1] #if value of smoker column is 1
print("The FEV of nonsmoker", FEV_non_smoker)
print("The FEV of nonsmokers", FEV_smoker)

#extract agelist of smokers and nonsmokers based on the values 0 and 1
age_nonsmoker= age[smoker_c ==0]   
age_smoker = age[smoker_c ==1]
print("age_nonsmoker", age_nonsmoker)
print("age_smoker", age_smoker)

#histogram of age vs nonsmokers
plt.hist(age_nonsmoker, 20, label='non_smokers')
plt.xlabel('age group of smokers and nonsmokers')
plt.ylabel('Count')
plt.title('Histogram over age group of nonsmokers')
plt.legend()
plt.show()

#histogram of agegroup vs smokers over FEV values

plt.hist(age_smoker,20, label='smokers')
plt.xlabel('age group of smokers')
plt.ylabel('Count')
plt.title('Histogram over age group over smokers')
plt.legend()
plt.show()



