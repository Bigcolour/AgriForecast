import json
import pandas as pd
import os

# 读取JSON文件
file_path = '/Users/bigcolour/Desktop/论文桌面整理/水产品价格预测/代码/new/1_数据收集/结果/response_2026-02-05_00-16-07.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取大黄鱼价格数据
price_data = data['Data']['Data']

# 整理数据
processed_data = []
for item in price_data:
    # 提取日期和价格
    date = item['dynamic_date'].split(' ')[0]  # 只保留日期部分
    market_price = item['market_price']
    vip_price = item['vip_price']
    
    # 添加到处理后的数据
    processed_data.append({
        '日期': date,
        '市场价格': market_price,
        'VIP价格': vip_price
    })

# 创建DataFrame
df = pd.DataFrame(processed_data)

# 保存为Excel文件
output_path = '/Users/bigcolour/Desktop/论文桌面整理/水产品价格预测/代码/new/1_数据收集/结果/大黄鱼价格数据.xlsx'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_excel(output_path, index=False)

print(f'数据已成功保存到: {output_path}')
print(f'共处理了 {len(processed_data)} 条数据')