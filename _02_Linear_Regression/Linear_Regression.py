# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
#对数据进行处理(标准化操作并将构建增广矩阵)
def data_processing_st(X):
    X = np.apply_along_axis(standard_st, 1, X)
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones))
    return X
#使用01规范法处理数据
def data_processing_01(X):

    X = np.apply_along_axis(standard_01, 1, X)
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones))
    return X
#使用01规范法处理数据
def standard_01(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data
#对测试数据进行统一标准化
def standard_st(data):
    data_mean = np.mean(data)
    # X_var=np.var(X)#方差
    data_std = np.std(data)
    data = (data - data_mean) / data_std
    return data
def ridge(data):
    data=standard_01(data)
    X, y = read_data()
    X = data_processing_01(X)
    # 选择模型并初始化参数
    w = np.zeros((7, 1))
    y_pre = X @ w  # 计算预测值
    # 定义损失函数
    # alpha = 10  # 正则化系数
    alpha=1#2023/4/30
    ridgeloss = 2 * (X.T @ X @ w - X.T @ y + alpha * w)
    # 对损失函数中w求偏导,令导数为0，求得w
    w = np.linalg.inv((X.T @ X + alpha * np.eye(np.shape((X.T @ X))[0]))) @ X.T @ y
    #获取w和b
    b = w[-1]
    w = w[:-1]
    w = w.reshape(6, 1)
    return data@w+b

    
def lasso(data):
    data = standard_st(data)
    X, y = read_data()
    X = data_processing_st(X)
    y=y.reshape(1,404)
    # 设置超参数
    alpha = 1000  # 正则化系数
    beta = 0.00045  # 学习率

    # 选择模型并初始化参数
    w = np.zeros((7, 1))
    best = w
    min = 365194055
    loss_old = 1  # 记录上一次的损失，如果两个损失变化太小，说明已经趋近最优值，提前停止
    # print(X@w)
    for i in range(100000):
        y_pre = X @ w  # 计算预测值
        # 定义损失函数
        mse = np.sum(((X @ w )- y.T) @ ((X @ w) - y.T).T)/(np.shape(X)[0])
        l1 = alpha * ((np.sum(np.abs(w))))
        lassoloss = mse + l1
        # 计算梯度
        dw = X.T @ ((X @ w) - y.T) + alpha * np.sign(w)  # 不是很理解为什么
        loss_old = lassoloss#记录上一次的风险损失
        # 更新参数
        w = w - beta * dw
        # 后边损失函数下降十分缓慢，设置提前停止的条件
        if (np.abs(min - loss_old) < 0.0001):
            print('提前停止！')
            break
        # 获取最小损失时候的参数w
        if (min >= lassoloss):
            min = lassoloss
            best = w
        # 输出损失函数的值

    #print(f'Iteration {i}: Loss = {lassoloss} ')
    w =best[0:6,:]
    b=best[6,0]
    print(data@w+b)
    return data@w+b

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y