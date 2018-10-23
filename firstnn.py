#coding=utf-8
import  numpy as np  #导入唯一的包

def sigmoid(x, deriv=False):  #定义激活函数，这里可以换成其他激活函数？例如tanh/relu
    if(deriv==True):
        return x*(1-x)  #返回导数值
    else:
        return 1/(1+np.exp(-x)) #返回sigmoid值



X=np.array([ [0,0,1],
             [1,1,1],
             [1,0,1],
             [0,1,1]

])    #训练集，这里是个4*3的数据

Y = np.array([[0,1,1,0]]).T  #标签

np.random.random(1)   #随机种子

syn0 = np.random.random([3,1])  #生成一个随机的3*1的向量，作为初始化的权重

for i in range(100000):   #设置循环次数，这里次数越大，权重训练的越好

    l0 = X

    y = Y

    l1 = sigmoid(np.dot(X,syn0))
    '''
      利用dot函数用训练集乘以初始化的权重，得到一个4*1的向量，然后用sigmoid函数将网络输出做非线性化处理
    '''
    l1_error = y-l1
    '''
      计算出误差值
    '''
    l1_delta = l1_error*sigmoid(l1,True)

    '''
      当l1较大或者较小时，计算出此时输出在激活函数导函数上对应的值，这个值一定很小
      
    '''

    syn0 += np.dot(l0.T,l1_delta)

    '''
      调整参数
    '''

l2=np.array([0,0,1])  #测试数据
l3=sigmoid(np.dot(l2,syn0))  #利用训练出来的数据对测试数据进行预测

print '网络最终的输出为：'

print l1

print '针对测试集进行预测，结果为'

print l3




