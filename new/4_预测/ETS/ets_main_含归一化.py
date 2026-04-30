import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.io import savemat
import scipy.io as sio
import numpy as np


# 读取数据集
df = pd.read_excel('4_预测/ETS/in.xlsx')

# 划分训练集和测试集 
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# 创建MinMaxScaler对象，并对训练和测试数据进行归一化
scaler = MinMaxScaler()
train_norm = scaler.fit_transform(train)
test_norm = scaler.transform(test)
# 实例化ETS模型并拟合训练数据
model = ExponentialSmoothing(train_norm, trend='add', seasonal_periods= 92)
model_fit = model.fit()
n_steps = len(test_norm)
y_hat_norm = model_fit.forecast(steps=n_steps)

# 对预测结果进行反归一化，得到真实值
y_hat_norm = y_hat_norm.reshape(-1, 1)
# print(y_hat_norm)
y_hat = scaler.inverse_transform(y_hat_norm)

# 将预测结果保存到MAT文件中
# y_hat.to_excel('4_预测/ETS/结果/ets_out.xlsx')
pd.DataFrame(y_hat).to_excel('4_预测/ETS/结果/ets_out.xlsx')

# 绘制预测结果和实际值的可视化图表
# plt.plot(train.index, train.values, label='Train')
# plt.plot(test.index, test.values, label='Test')
# plt.plot(y_hat.index, y_hat.values, label='Predicted')
# plt.legend()
# plt.savefig('figure/ETS预测结果.jpg')
# plt.show()

# 计算均方误差（MSE）来评估模型预测性能
mse = mean_squared_error(test, y_hat)
print(f'Mean Squared Error: {mse}')
