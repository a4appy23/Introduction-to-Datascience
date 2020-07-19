import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy.linalg as linalg


cell = np.loadtxt('diatoms.txt')
#center the data

mean = np.mean(cell, axis=0)
data_scaled= cell-mean

dataset_mat = np.cov(data_scaled)   
eigenValues, eigenVectors = linalg.eig(dataset_mat)

idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]




mean1 = np.mean(cell, axis=1)
e1 = eigenVectors[:,0]
std1 = np.sqrt(eigenValues[0])
#creating list for cells
xlist=[]
ylist=[]

def plot(data, **kwargs):
    j=0
    for j in range(5):
        cells=cell[j]

        for index, i in enumerate(cells):
            if index % 2 == 0:
                xlist.append(i)
        
            else:
                ylist.append(i)
            #print(i)
                index += 1
        plt.plot(xlist,ylist)
        plt.title('plot of 5 cells with pc')
plot(cell)     
plt.show()         
plt.axis('equal')
      

 #plot of 5 cells with colormap and pc
blues = plt.get_cmap('Blues') 
#plotting first PC
for i in range (-2,3):#(including -2,-1,0,1,2)
    plots1 = mean1 + i*std1*e1
    plot(plots1,color=blues)
    plt.title('plot of 5 cells with 1st pc') 
    plt.axis('equal')
    plt.show()
    
#plotting second PC
mean2 = np.mean(cell, axis=1)
e2 = eigenVectors[:,1]
std2 = np.sqrt(eigenValues[1])
for i in range (-2,3):
    plots2 = mean2 + i*std2*e2
   # plot(plots2,color=blues)
   # plt.title('plot of 5 cells with 2st pc')
    #plt.axis('equal')
    #plt.show()
#plotting second PC
mean3 = np.mean(cell, axis=1)
e3 = eigenVectors[:,2]
std3 = np.sqrt(eigenValues[2])

for i in range (-2,3):
    plots3 = mean3 + i*std3*e3
   # plot(plots3,color=blues)
    
    #plt.title('plot of 5 cells with 3st pc')
    #plt.axis('equal')
    #plt.show()











