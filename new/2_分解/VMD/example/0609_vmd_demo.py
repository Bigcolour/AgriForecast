import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
import datetime

# ---------如何确定模态分解个数？
# ---分解后，看各个模态中心频率，一般中心频率之间相差一倍以上（中高低），对预测效果较好


# -----测试信号及其参数--start-------------
# T=1148;fs=1/T;t=np.arange(1,T+1)/T
df = pd.read_excel('2_分解/data_2948.xlsx')
f = df['price'].values
T = len(f)
fs = 1/T
t = np.arange(1, T+1)/T

# -----测试信号及其参数--end----------
alpha = 3000  # alpha 带宽限制经验取值为抽样点长度1.5-2.0倍
tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
K = 6  # K 分解模态（IMF）个数
DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
tol = 1e-6  # 控制误差大小常量，决定精度与迭代次数
# 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
# 画原始信号和它的各成分

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output = pd.DataFrame(u)
output.to_excel(f'2_分解/VMD/结果/vmd_result_{timestamp}.xlsx')

plt.figure(figsize=(10, 7))
plt.subplot(K+1, 1, 1)
plt.plot(t, f)
plt.suptitle('Original input signal and its components')  # 原始输入信号
plt.show()
# 分解出来的各IMF分量
plt.figure(figsize=(10, 7))
plt.plot(t, u.T)
plt.title('all Decomposed modes')
plt.show()  # u.T是对u的转置


imfs = u

# 分别绘制每个IMF分量
num_imfs = imfs.shape[0]

fig, axs = plt.subplots(num_imfs, 1, figsize=(8, 1.5*num_imfs))

for i, ax in enumerate(axs):
    ax.plot(t, imfs[i])
    ax.set_ylabel(f'IMF {i+1}')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Amplitude')

plt.tight_layout()
plt.show()

residual = f - np.sum(imfs, axis=0)
plt.figure(figsize=(10, 7))
plt.plot(t, residual)
plt.title('Residual signal')
plt.show()
