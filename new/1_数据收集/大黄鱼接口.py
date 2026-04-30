#!/usr/bin/env python3
import requests
import json
import time
import logging
import datetime

# 格式化时间
# localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# 设定一些变量值
url = 'https://nddhy.org315.cn/api/Column/marketquotation'
header = {
	'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.13(0x18000d29) NetType/WIFI Language/zh_CN',
	'Accept': 'application/json, text/javascript, */*; q=0.01'
}


# c_type_id=4&pageIndex=1&pageSize=8&token=a07a661c08b98a145dfbcf9ac0cbd6ce'


# cookie2 为电脑端通过统一认证得到的
# cookie_pc = {
# "wechatSESS_ID": "ffc82592b29af5ca72bab64f6c28bf7f27924d1b4ada5ffa",
# "Hm_lpvt_7ecd21a13263a714793f376c18038a87": "1631639466",
# "Hm_lvt_7ecd21a13263a714793f376c18038a87": "1631511833,1631541541,1631550829,1631638507",
# "FROM_CODE": "WwsBDFQC",
# "FROM_TYPE": "weixin",
# "gench_hq_user": "1amC9Ns9qegnRL6GHXYvA=="
# }
result = requests.post(url=url, data={
                       "c_type_id": "4",
                       "pageIndex": "1",
                       "pageSize": "999",
                       "token": "a07a661c08b98a145dfbcf9ac0cbd6ce"}, headers=header)
print(result.text)


if result.status_code == 200:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"1_数据收集/结果/response_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(result.text)
else:
    print(f"Request failed with status code {result.status_code}")


# ticks = datetime.datetime.now()
# print(ticks)
# aa = str(ticks) + result.text
# with open("test.txt","w") as f:
# f.write(aa)  # 自带文件关闭功能，不需要再写f.close()
