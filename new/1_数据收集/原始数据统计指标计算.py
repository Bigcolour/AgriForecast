import pandas as pd
from scipy.stats import skew, kurtosis

# 读取 csv 文件并将其中的日期作为索引
df = pd.read_excel('2_分解/data_3248.xlsx')
f = df['price'].values
print(f)
# 计算每个时间窗口内的平均值、最大值、最小值、中位数、标准差、偏度和峰度
# resampled = df.resample('1D')  # 将数据重采样为每天
statistics = df.agg(['mean', 'max', 'min', 'median', 'std','skew', 'kurtosis'])

# 打印结果
print(statistics)
