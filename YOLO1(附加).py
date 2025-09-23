#左图红点为原始数据，蓝线为拟合函数，系数已输出
#右图为损失函数
import matplotlib.pyplot as plt
import numpy as np
rate=0.03
w=[1,1,1,1,1]
def fun(x):
    return 2*x**5 + x**4 - 11*x**3 + 7*x**2 + 2*x
def estimate(x,w):
    return w[0]*x**5 + w[1]*x**4 + w[2]*x**3 + w[3]*x**2 + w[4]*x
def loss(w):
    tot=0
    for i in numx:
        tot+=(fun(i)-estimate(i,w))**2
    tot/=40
    return tot
def loss_dao(i):
    s=0
    for k in range(20):
        s+=(estimate(numx[k],w) - numy[k]) * numx[k]**(5-i)  
    return s/20

oldw=[[],[],[],[],[]]
def gradient():
    for j in range (5):
        oldw[j].append(w[j])
        w[j]=w[j]-rate*loss_dao(j) 
    

numx=[]
numy=[]
i=-0.5
for j in range (20):
    numx.append(i)
    numy.append(fun(i))
    i+=0.1
q=loss(w)
history=[q]
while q>=0.001:
    gradient()
    q=loss(w)
    history.append(q)
print(w)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("function")
plt.plot(numx,numy,'or')
x=np.linspace(-0.5,1.5)
y=estimate(x,w)
plt.plot(x,y)

plt.subplot(1,2,2)
plt.title("loss")
plt.plot(history)
plt.show()
