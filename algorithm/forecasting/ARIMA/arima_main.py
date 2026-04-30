import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error

# 读取数据集
# df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', 
#                  header=0, index_col=0, parse_dates=True)
df = pd.read_excel('4_预测/ARIMA/in.xlsx')


# 划分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# 实例化ARIMA模型并拟合训练数据
model = ARIMA(train, order=(1, 1, 3))
model_fit = model.fit()

# 进行预测并输出结果
n_steps = len(test)
y_hat = model_fit.predict(start=train_size, end=train_size+n_steps-1)
# np.savetxt('4_预测/ARIMA/arima_main_result.txt', y_hat, delimiter='\n')
y_hat.to_excel('4_预测/ARIMA/arima_main_result.xlsx')

# 绘制预测结果和实际值的可视化图表
# plt.plot(train.index, train.values, label='Train')
# plt.plot(test.index, test.values, label='Test')
# plt.plot(y_hat.index, y_hat.values, label='Predicted')
# plt.legend()
# plt.show()

# 计算均方误差（MSE）来评估模型预测性能
mse = mean_squared_error(test, y_hat)
print(f'Mean Squared Error: {mse:.2f}')
