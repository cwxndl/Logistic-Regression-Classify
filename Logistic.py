import numpy as np 
from numpy import *
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report

class MyLogistic():
    ## 初始化参数
    def __init__(self,learning_rate,iter_nums,Isdynamic,Algorithm):
        self.learning_rate = learning_rate  ##学习速率
        self.iter_nums = iter_nums          ##迭代次数   
        self.Isdynamic = Isdynamic          ##用户自己选择是否画动态图
        self.Algorithm = Algorithm          ##用户选择自己需要的优化算法    
    ## 为了防止计算机计算溢出，需要分情况定义sigmiod函数
    def sigmoid(self,x):
        if x>=0:
            return 1.0/(1+np.exp(-x)) 
        else:
            return np.exp(x)/(1+np.exp(x))   
    ## 计算梯度矩阵
    def gradient(self,x,y,beta):
        shape = x.shape #获取数据的行数和列数
        rows = shape[0] #数据的行数
        cols = shape[1] #数据特征维数
        diff_matrix = np.zeros((1,cols)) #初始化数据集的梯度矩阵
        for  i in range(rows): # 遍历每一个样本数据
            temp_x = x[i,:] # 读取第i个样本数据
            temp_y = y[i]   # 读取第i个样本标签
            p1 = self.sigmoid(np.dot(beta,temp_x.reshape(cols,1))) #计算该样本属于正例的概率
            diff_matrix = np.add(diff_matrix,temp_x*(p1-temp_y)) #更新梯度矩阵
        return  diff_matrix  #返回梯度矩阵
    
    ## 定义经典的梯度下降算法
    def gradient_descent(self,x,y,beta,iters):
        loss =[]
        ## 统计运行时间
        import time 
        start = time.time()
        for i in range(iters):
            diff_matrix = self.gradient(x,y,beta) ##使用梯度下降算法进行更新beta 和 diff_mat
            beta = beta - self.learning_rate*diff_matrix
            loss.append(self.loss(x=x,y=y,beta=beta))
        end = time.time()
        print('使用经典梯度下降算法运行时间为：{:5f}'.format(end-start))
        return beta,loss
    
    ## 定义随机梯度下降算法
    def SGD(self,x,y,beta,iters):
        loss = []
        import random
        random.seed(2)
        ## 统计运行时间
        import time 
        start = time.time()
        for iter in range(iters):
            random_index = np.random.randint(0,x.shape[0],1)   ##随机选择一个样本进行梯度下降
            temp_x = x[random_index,:] ##随机选择的样本特征
            temp_y = y[random_index]   ##随机选择的样本标签
            diff_matrix = self.gradient(temp_x,temp_y,beta)  ##计算梯度（此时只有一个样本，就是一个行向量）
            beta = beta - self.learning_rate*0.5*diff_matrix ##更新beta，一般来说随机梯度下降算法的步长要比批量梯度下降算法的步长要小
            loss.append(self.loss(x=x,y=y,beta = beta))
        end = time.time()
        print("使用随机梯度下降算法运行时间为：{:5f}".format(end-start))
        return beta,loss

    ## 定义损失函数
    def loss(self,x,y,beta):
        raws,cols = x.shape
        temp_y = np.dot(beta,x.transpose()) ## 
        loss_sum = 0
        for i in range(raws):
            if self.sigmoid(temp_y[0,i])>=0.5:
                loss = 1/raws*np.power((1-y[i]),2)
            else:
                loss = 1/raws*np.power(y[i],2)
            loss_sum = loss_sum+loss
        return loss_sum
    ## 定义绘图函数  实际上动态绘图具有很大的时间成本，你也可以选择绘制最终的图形
    def plot_loss(self,loss):
        import matplotlib.pyplot as plt
        if self.Isdynamic:
            plt.ion()# 打开交互模式
            plt.figure(figsize=(5,3.5),dpi=150)
            for i in range(self.iter_nums):
                x = list(range(1, self.iter_nums+1)) 
                ix = x[:i]
                iy = loss[:i]
                plt.clf()
                plt.cla()
                plt.plot(ix,iy,'r',label = 'Loss value on traindata')
                plt.legend(loc = 'upper right')
                plt.rcParams['font.family'] = 'Microsoft YaHei'  # 设置微软雅黑字体
                plt.tick_params(axis='both',direction ='in')
                plt.xlabel('Iter')
                plt.ylabel('Loss')
                plt.grid(False)
                plt.pause(0.05)
            plt.ioff() # 关闭交互模式
            plt.show()
        else:
            plt.figure(figsize=(5,3.5),dpi=150)
            plt.plot(loss,'r',label = 'Loss value on traindata')
            plt.legend(loc = 'upper right')
            plt.rcParams['font.family'] = 'Microsoft YaHei'  # 设置微软雅黑字体
            plt.tick_params(axis='both',direction ='in')
            plt.xlabel('Iter')
            plt.ylabel('Loss')
            plt.grid(False)
            plt.show()


    ## 定义训练函数fit
    def fit(self,X_train,y_train):
        import matplotlib.pyplot as plt
        # beta = np.zeros((1,X_train.shape[1]))  ##初始化参数
        self.beta = np.zeros((1,X_train.shape[1]))
        if self.Algorithm =='GD':
            # beta,loss = self.gradient_descent(x=X_train,y=y_train,beta=beta,iters=self.iter_nums)
            self.beta,loss = self.gradient_descent(x=X_train,y=y_train,beta=self.beta,iters=self.iter_nums)
        else:       
            # beta,loss = self.SGD(x=X_train,y=y_train,beta=beta,iters=self.iter_nums)
            self.beta,loss = self.SGD(x=X_train,y=y_train,beta=self.beta,iters=self.iter_nums)
        self.plot_loss(loss=loss)
        return self.beta    
    
    ## 定义预测函数
    # def predict(self,beta,x):
    def predict(self,x):
        # temp_y = np.dot(beta,np.transpose(x))
        temp_y = np.dot(self.beta,np.transpose(x))
        predict_y = []
        for i in range(temp_y.shape[1]):
            y = self.sigmoid(temp_y[0,i])
            if y>=0.5:
                predict_y.append(1)
            else:
                predict_y.append(0)
        return predict_y

