import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('评价指标计算/预测汇总-2948-202306-原版.xlsx')
# dff = pd.read_excel('5_权重分配/输出/output_ETS.xlsx')

# 读取第一列为预测值，第二列为实际值
y_true = df.iloc[:, 1].values.tolist()
y_pred = df.iloc[:, 13].values.tolist()
# y_pred = df.iloc[:, 0].values.tolist()
# y_true = df.iloc[:, 1].values.tolist()

# print(y_true)
# print(y_pred)

# 计算MSE
mse = np.mean((np.array(y_pred) - np.array(y_true))**2)

# # 计算RMSE
rmse = np.sqrt(mse)

# # 计算MAE
mae = np.mean(np.abs(np.array(y_pred) - np.array(y_true)))

# # 计算MAPE
mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

# 计算SMAPE
smape = 100 * np.mean(2 * np.abs(np.array(y_true) - np.array(y_pred)) / (np.abs(np.array(y_true)) + np.abs(np.array(y_pred))))

# 输出结果
print(mse)
print(rmse)
print(mae)
print(mape)
print(smape)