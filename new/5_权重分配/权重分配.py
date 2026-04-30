import pandas as pd

# 读取指定工作表的Excel文件
sheet_name = 'FULL'  # 替换成您要读取的工作表名称
df = pd.read_excel('5_权重分配/new_larimichthysdail_0630.xlsx', sheet_name=sheet_name)

# 手动输入权重
result = [0.17485,0.0001,0.29719,0.0001,0.0001,0.18904,0.0001,0.079839,0.22146]
divisor = 0.962779  # 要除以的数

weights = [value / divisor for value in result]

# 计算加权结果
# df['Weighted Result'] = df.iloc[:, 0] * weights[0] + df.iloc[:, 1] * weights[1] + df.iloc[:, 2] * weights[2]
df['Weighted Result'] = df.iloc[:, 0] * weights[0] + df.iloc[:, 1] * weights[1] + df.iloc[:, 2] * weights[2] + df.iloc[:, 3] * weights[3] + df.iloc[:, 4] * weights[4] + df.iloc[:, 5] * weights[5] + df.iloc[:, 6] * weights[6] + df.iloc[:, 7] * weights[7] + df.iloc[:, 8] * weights[8]

# 保存结果到新的Excel文件
df['Weighted Result'].to_excel(f'5_权重分配/输出/output_{sheet_name}.xlsx', sheet_name=sheet_name, index=False)

