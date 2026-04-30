import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.io import savemat
import scipy.io as sio
import numpy as np

# 读取数据集
df = pd.read_excel('4_预测/ETS-202306/in.xlsx')

# 划分训练集和测试集 
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# 实例化三次指数平滑（Holt-Winters）模型并拟合训练数据
model = ExponentialSmoothing(train, trend='add', seasonal='add', damped_trend=True,
                             seasonal_periods=365)
model_fit = model.fit()
n_steps = len(test)
y_hat = model_fit.forecast(steps=n_steps)

# 将预测结果保存到txt文件中
np.savetxt('4_预测/ETS-202306/结果/ets_out.txt', y_hat)

# 计算均方误差（MSE）、均方根误差（RMSE）、平均绝对误差（MAE）、决定系数（R2）来评估模型预测性能
mse = mean_squared_error(test, y_hat)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, y_hat)
r2 = r2_score(test, y_hat)

print(f'mse: {mse}')
print(f'rmse: {rmse}')
print(f'mae: {mae}')
print(f'R2 Score: {r2}')

# 绘制预测结果和实际值的可视化图表
# plt.plot(train.index, train.values, label='Train')
# plt.plot(test.index, test.values, label='Test')
# plt.plot(test.index, y_hat, label='Predicted')
# plt.legend()
# plt.savefig('figure/ETS预测结果.jpg')
# plt.show()
