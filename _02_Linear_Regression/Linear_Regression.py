# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
#对数据进行处理(标准化操作并将构建增广矩阵)
def data_processing(X):
    # 处理下数据
    # y = y.reshape(1, 404)
    X_mean = np.mean(X)
    # X_var=np.var(X)#方差
    X_std = np.std(X)
    X = (X - X_mean) / X_std
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones))
    return X
def ridge(data):
    X, y = read_data()
    X = data_processing(X)
    # 选择模型并初始化参数
    w = np.zeros((7, 1))
    y_pre = X @ w  # 计算预测值
    # 定义损失函数
    alpha = 10  # 正则化系数
    ridgeloss = 2 * (X.T @ X @ w - X.T @ y + alpha * w)
    # 对损失函数中w求偏导,令导数为0，求得w
    w = np.linalg.inv((X.T @ X + alpha * np.eye(np.shape((X.T @ X))[0]))) @ X.T @ y
    #获取w和b
    b = w[-1]
    w = w[:-1]
    w = w.reshape(6, 1)
    return data@w+b

    
def lasso(data):
    X, y = read_data()
    X = data_processing(X)
    # 设置超参数
    alpha = 1  # 正则化系数
    beta = 0.0001  # 学习率

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
        if (np.abs(min - loss_old) < 0.001):
            print('提前停止！')
            break
        # 获取最小损失时候的参数w
        if (min >= lassoloss):
            min = lassoloss
            best = w
        # 输出损失函数的值

        print(f'Iteration {i}: Loss = {lassoloss} ')
        w =best[0:6,:]
        b=best[6,0]
        return data@w+b

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y