import pandas as pd
from datetime import timedelta, date

start_date = date(2015, 5, 13)
end_date = date(2024, 3, 12)

date_range = []

for n in range(int((end_date - start_date).days)+1):
    date_range.append(start_date + timedelta(n))

df = pd.DataFrame({'Date': date_range})
df.to_excel('1_数据收集/dates.xlsx', index=False)
