# -*- coding: utf-8 -*-
from sklearn.svm import SVR
from math import sqrt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import ELM
import pandas as pd
from scipy.io import savemat
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
np.random.seed(0)

from pylab import mpl
 
# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def split_data(data, n):
    in_ = []
    out_ = []
    N = data.shape[0] - n
    for i in range(N):
        in_.append(data[i:i + n,:])
        out_.append(data[i + n,:])
    in_ = np.array(in_).reshape(len(in_), -1)
    out_ = np.array(out_).reshape(len(out_), -1)
    return in_, out_


# In[] 加载数据
data=pd.read_excel('4_预测/ELM-202306/in.xlsx') .values[:,:]

step=3
x,y=split_data(data,step)

m=int(0.8*x.shape[0])
X_train=x[:m,]
X_test=x[m:,]
y_train=y[:m,]
y_test=y[m:,]# 归一化或标准化
ss_x=StandardScaler().fit(X_train)
ss_y=StandardScaler().fit(y_train)
# ss_x=MinMaxScaler().fit(X_train)
# ss_y=MinMaxScaler().fit(y_train)

train_data=ss_x.transform(X_train)
train_label =ss_y.transform(y_train)
test_data=ss_x.transform(X_test)
test_label=ss_y.transform(y_test)



# 建模
N=50
Model = ELM.ELMBase(train_data,N)
Model.regressor_train(train_label)
test_pred=Model.regressor_test(test_data)
# In[] 画出测试集的值
# 对测试结果进行反归一化
test_label=test_label.reshape(-1,1)
inv_y = ss_y.inverse_transform(test_label)
test_pred=test_pred.reshape(-1,1)
inv_yhat  = ss_y.inverse_transform(test_pred)


#保存结果
# print(inv_yhat,'preprepre')
# savemat('4_预测/ELM/结果/elm_result.mat',{'true':inv_y,'pred':inv_yhat})
np.savetxt('4_预测/ELM-202306/结果txt/elm_result.txt', inv_yhat, delimiter='\n')
# In[]计算各种指标
rmse = sqrt(mean_squared_error(inv_y, inv_yhat ))
mse = mean_squared_error(inv_y, inv_yhat )
print(f'Mean Squared Error: {mse:.8f}')
print('Test RMSE: %.7f' % rmse)
print('Test MAE: %.7f' % mean_absolute_error(inv_y, inv_yhat))
print('Test R2: %.7f' % r2_score(inv_y, inv_yhat))



# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['Kaitt', 'PingFang HK']
 
# 正常显示负号
# plt.rcParams['axes.unicode_minus'] = False
# # plot test_set result
# plt.figure()
# plt.plot(inv_y, c='r', label='real')
# plt.plot(inv_yhat, c='b', label='pred')
# plt.legend()
# plt.xlabel('样本点')
# plt.ylabel('功率')
# plt.savefig('figure/ELM预测结果.jpg')
# plt.show()

