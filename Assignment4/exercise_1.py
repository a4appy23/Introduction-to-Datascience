from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

cell= np.loadtxt('diatoms.txt')
print(cell.shape)

#plot of one cell
cell1 =cell[0]
xlist=[]
ylist=[]

for index, i in enumerate(cell1):
    if index % 2 == 0:
        xlist.append(i)
        
    else:
        ylist.append(i)
        #print(i)
  #  index += 1
print(len(xlist)) 
plt.plot(xlist,ylist) 
plt.axis('equal')
plt.title('plot of one cell')
plt.show()
#plot of many cell


j=0
for j in range(780):
    cells=cell[j]

    for index, i in enumerate(cells):
        if index % 2 == 0:
            xlist.append(i)
        
        else:
            ylist.append(i)
            #print(i)
         #  index += 1
       
plt.plot(xlist,ylist) 
plt.axis('equal')
plt.title('plot of many cell')
plt.show()       

    

    
    







 
   
    