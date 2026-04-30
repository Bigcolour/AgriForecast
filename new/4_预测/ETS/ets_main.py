import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from scipy.io import savemat
import scipy.io as sio
import numpy as np


# 读取数据集
# df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', 
#                  header=0, index_col=0, parse_dates=True)
# df=pd.read_excel('数据.xlsx') .values[:,:]
df = pd.read_excel('4_预测/ETS/in.xlsx')

# print(df) 

# 划分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(train)

# 实例化ETS模型并拟合训练数据
model = ExponentialSmoothing(train, trend='add', seasonal_periods=365)
# 1148：264
# vmd 分量一：55 分量二：分量三：14 分量四：分量五：分量六：
# 示例使用了additive趋势和季节性模型，并且将季节周期设置为12，
# 因为AirPassengers数据集是按照月份分组的数据。如果您使用的是其他时间周期的数据集，就需要调整相关参数。
model_fit = model.fit()

# 进行预测并输出结果
n_steps = len(test)
y_hat = model_fit.forecast(steps=n_steps)
print(y_hat)


# 将预测结果保存到MAT文件中
# sio.savemat('ets_result.mat', {'predicted': y_hat})
# np.savetxt('4_预测/ARIMA/结果txt/ets_result.txt', y_hat, delimiter='\n')
y_hat.to_excel('4_预测/ETS/结果/ets_out.xlsx')
# temp = pd.DataFrame(y_hat,index=None, columns=None)
# print(temp,'preprepre')
# savemat('结果/ets_result.mat',{'true':test,'pred':y_hat})


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
