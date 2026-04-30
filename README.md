# 数据预测网站

一个前后端分离的时间序列预测网站。前端使用 Vite + React + HeroUI，后端使用 Python + FastAPI，用户可以上传 CSV/XLSX 数据、选择预测模型、填写参数、提交服务器计算并查看预测结果。

## 功能

- 上传 CSV/XLSX 或加载示例数据。
- 前端预览 CSV 字段、行数、时间范围、数值范围和前几行数据；XLSX 由后端解析首个工作表。
- 可在预测前启用 VMD、SSA、EWT 数据分解，把原始序列拆成多个子序列分别预测，再用加法模型组合成最终预测。
- 分解参数和预测模型参数都支持自动寻优，通过时间顺序回测误差选择更合适的组合。
- 选择预测模型、预测步数和模型参数。
- 支持波动保留：把历史短期波动模式加入未来预测，避免结果过度平滑。
- 后端异步执行预测任务，前端自动轮询任务状态。
- 使用折线图展示历史数据和预测数据。
- 导出预测结果 CSV。

## 项目结构

```text
.
├── backend
│   ├── app
│   │   ├── main.py
│   │   ├── models.py
│   │   └── forecasting.py
│   └── requirements.txt
└── frontend
    ├── src
    │   ├── App.tsx
    │   ├── api.ts
    │   ├── main.tsx
    │   └── styles.css
    └── package.json
```

## 数据格式

上传 CSV 或 XLSX 文件，至少包含两列：

```csv
date,value
2025-01-01,120
2025-01-02,132
2025-01-03,128
```

- `date`：时间列，支持常见日期格式。
- `value`：数值列，必须可以转换为数字。
- CSV 支持 UTF-8 和常见的 GB18030 中文编码。
- XLSX 会读取首个工作表，表头同样使用 `date/time/ds` 和 `value/y/target` 这类字段名。

## 启动后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认前端地址是 `http://localhost:5173`，后端地址是 `http://localhost:8000`。

## 已实现模型

### 分解模型

- 不分解：直接使用原始序列预测。
- VMD 变分模态分解：将序列拆成多个 IMF 分量，并保留残差分量。
- SSA 奇异谱分析：通过轨迹矩阵和 SVD 重构多个趋势/振荡分量，并保留残差分量。
- EWT 经验小波分解：按频谱局部峰值划分频带，用 Meyer 滤波器提取子序列，并保留残差分量。

### 预测模型

- 朴素预测：使用最后一个观测值作为未来预测。
- 移动平均：使用最近 `window` 个观测值的均值预测未来。
- 简单指数平滑：使用 `alpha` 控制近期数据权重。
- 线性趋势：拟合一条线性趋势并向未来延伸。
- ETS 指数平滑：使用水平、趋势和可选季节项生成预测。
- ELM 极限学习机：使用历史窗口、随机隐藏层和岭回归进行非线性递推预测。
- LSTM 循环网络：使用轻量 LSTM 风格门控循环单元编码历史窗口，并训练输出层预测未来。
