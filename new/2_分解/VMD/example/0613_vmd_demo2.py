import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD
import datetime

# 从Excel文件读取数据
df = pd.read_excel('2_分解/data_3228.xlsx')
f = df['Price'].values
T = len(f)
fs = 1/T
t = np.arange(1, T+1)/T

# 设置VMD参数
alpha = 3000  # alpha 带宽限制经验取值为抽样点长度1.5-2.0倍
tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
K = 5  # K 分解模态（IMF）个数
DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
tol = 1e-6  # 控制误差大小常量，决定精度与迭代次数
# 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)


# 绘制结果
fig, ax = plt.subplots(nrows=K+1, ncols=1, figsize=(12,14))

for i in range(u.shape[0]):
    ax[i].plot(u[i], label=f'IMF {i+1}', color='blue')
    ax[i].legend()

ax[K].plot(f-np.sum(u, axis=0), label='Residual', color='red')
ax[K].legend()

plt.show()


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
IMF_data = {'IMF{}'.format(i+1): u[i] for i in range(K)}
IMF_data['Residual'] = f - np.sum(u, axis=0)
df_imf = pd.DataFrame(IMF_data)
df_imf.to_excel(f'2_分解/VMD/结果/vmd_result_{timestamp}.xlsx')