#本代码能实现：绘制函数，随机开始梯度下降并在图中表示其路径。
#for me:matplotlib面向对象创建、点与点之间的连线,随机数生成 3个知识点
#摆了的：动态路径
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-0.5,1.5)
y= 2*x**5 + x**4 - 11*x**3 + 7*x**2 + 2*x

'''
plt.xlim(-0.5,1.5)
plt.ylim(-1,2)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)
plt.show()
'''

fig , ax = plt.subplots()
ax.set_xlim(-0.5,1.5)
ax.set_ylim(-1,2)
ax.plot(x,y)



x1=np.random.uniform(-0.5,1.5)
sxp=[x1]
rate=0.01
def fun(x):
    return 2*x**5 + x**4 - 11*x**3 + 7*x**2 + 2*x
def dao(x):
    return 10*x**4 + 4*x**3 - 33*x**2 + 14*x + 2

omg=[fun(x1)]
def gradient():
    x=sxp[-1]
    if x<-0.5 or x>1.5:
         return 0
    sxp.append(x-rate*dao(x))
    omg.append( fun(sxp[-1]) )
    
    if abs(omg[-1] - omg[-2] )>=0.001 :
           gradient();
gradient()
'''
point = ax.plot([], [], 'bo', markersize=12)[0]
trail = ax.plot([], [], 'b--', alpha=0.5)[0]  # 轨迹线
point.set_data(x_path[frame], y_path[frame])
trail.set_data(x_path[:frame+1], y_path[:frame+1])
animation = FuncAnimation(fig, update, frames=len(x_path),
                          init_func=init, blit=True, interval=30)
plt.show()
'''
xpoint=np.array(sxp)
ypoint=np.array(omg)
print(xpoint)

plt.plot(xpoint,ypoint,'bo')
plt.plot(xpoint,ypoint)
plt.show()
print("fine")