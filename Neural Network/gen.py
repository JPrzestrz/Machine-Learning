import numpy as np
x1 =np.zeros(10)
y1 =np.zeros(10)
for i in range(10):
    x1[i]=i
    if i<=1:
        y1[i] = 1.1
    elif i<=4:
        y1[i]=i+0.1
    else:
        y1[i]= 4.1

for i in range(10):
    print('['+str(x1[i])+','+str(y1[i])+']')