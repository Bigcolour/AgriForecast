#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# path = "BP、SVR、ELM三种常见单一对比模型-单输入/数据.txt"  # 数据集路径
df = pd.read_excel('2_分解/data_3228.xlsx')
f = df['Price'].values

series = df['Price'].values
series = series - np.mean(series)   # 中心化(非必须)

# step1 嵌入
windowLen = 6           # 嵌入窗口长度
seriesLen = len(series)     # 序列长度
print(seriesLen,'lenlen')
K = seriesLen - windowLen + 1
X = np.zeros((windowLen, K))
for i in range(K):
    X[:, i] = series[i:i + windowLen]

# step2: svd分解， U和sigma已经按升序排序
U, sigma, VT = np.linalg.svd(X, full_matrices=False)

for i in range(VT.shape[0]):
    VT[i, :] *= sigma[i]
A = VT

# 重组
rec = np.zeros((windowLen, seriesLen))
for i in range(windowLen):
    for j in range(windowLen-1):
        for m in range(j+1):
            rec[i, j] += A[i, j-m] * U[m, i]
        rec[i, j] /= (j+1)
    for j in range(windowLen-1, seriesLen - windowLen + 1):
        for m in range(windowLen):
            rec[i, j] += A[i, j-m] * U[m, i]
        rec[i, j] /= windowLen
    for j in range(seriesLen - windowLen + 1, seriesLen):
        for m in range(j-seriesLen+windowLen, windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (seriesLen - j)
        
rrr = np.sum(rec, axis=0)  # 选择重构的部分，这里选了全部

# 计算残差值
residual = series - rrr

# 将rec结果和残差值保存到excel中
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
df_rec = pd.DataFrame(rec.T, columns=[f"rec_{i}" for i in range(0, windowLen)])
# df_res = pd.DataFrame(residual, columns=['residual'])
df_all = df.join([df_rec])
df_all.to_excel(f"2_分解/SSA/结果/ssa_res_{timestamp}.xlsx", index=False)

# 绘制图像
plt.figure()
for i in range(windowLen):
    ax = plt.subplot(windowLen+1,1,i+1,)
    ax.plot(rec[i, :])

ax = plt.subplot(windowLen+1,1,windowLen+1)
ax.plot(residual, label='Residual', color='red')
ax.legend()
plt.show()
