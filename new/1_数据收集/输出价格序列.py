import json
from openpyxl import Workbook

# 从JSON文件加载数据
with open("1_数据收集/结果/response.json", "r") as f:
    response_data = json.load(f)

# 获取Data中的数据列表
data_list = response_data['Data']['Data']

# 创建一个工作簿和一个工作表
workbook = Workbook()
worksheet = workbook.active

# 写入数据标题行
worksheet.append(["dynamic_date", "vip_price"])

# 遍历数据列表，将其写入工作表
for data in data_list:
    row = [data["dynamic_date"], data["vip_price"]]
    worksheet.append(row)

# 保存工作簿到文件
workbook.save(filename="1_数据收集/date_and_vip_prices.xlsx")
