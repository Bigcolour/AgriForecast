# %% [markdown]
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
plt.rcParams['axes.unicode_minus']=False       #显示负号

# %%
data=pd.read_excel('4_预测/LSTM/分解数据/data_3228.xlsx',usecols=[0])

# %%
data

# %%
data.isnull()

# %%
True in data.isnull()

# %%
data.plot()

# %%
seq_test=data.values#读取预测的主要变量

# %%
seq_test.shape

# %%
origin=seq_test

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
sc=MinMaxScaler()#归一化数据
seq_test=sc.fit_transform(seq_test.reshape(-1,1))

# %%
def split_sequence(sequence, look_back):
    X, y = [], []
    for i in range(len(sequence)):
        # 找到最后一次滑动所截取数据中最后一个元素的索引，
        # 如果这个索引超过原序列中元素的索引则不截取；
        end_element_index = i + look_back
        if end_element_index > len(sequence) - 1: # 序列中最后一个元素的索引
            break
        sequence_x, sequence_y = sequence[i:end_element_index], sequence[end_element_index] # 取最后一个元素作为预测值y
        X.append(sequence_x)
        y.append(sequence_y[0])
    
    #return X,y
    return np.array(X), np.array(y)

# %%
look_back = 6#回顾历史多长时间
seq_test_x, seq_test_y = split_sequence(seq_test, look_back)#滑动窗口生成数据

# %%
#滑动窗口到Lstm的输入转换
seq_test_x_1 = seq_test_x.reshape((seq_test_x.shape[0], look_back, 1))#将二维数据变为3维

# %%
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,LSTM,Bidirectional
from keras.optimizers import Adam

# %%
model = Sequential()
model.add(Conv1D(filters=64,kernel_size=(2,),strides=1,padding='same',input_shape=(look_back,1)))
# model.add(LSTM(32,activation='relu'))#添加lstm层
model.add(Bidirectional(LSTM(32,activation='relu')))#添加bilstm层
#下面3个dense是全连接层
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# %%
model.compile(loss='mae', optimizer=Adam(.001), metrics=['mae'])

# %%
model.summary()

# %%
#模型可视化
from keras.utils.vis_utils import plot_model
plot_model(model,show_shapes=True)

# %%
len(data)

# %%
model.fit(x=seq_test_x_1[:2583],y=seq_test_y[:2583],batch_size=16,epochs=50,verbose=2)#模型训练。一共1148个数据，拿前918个做训练

# %%
len(seq_test_x_1)

# %%
yre=model.predict(seq_test_x_1)#模型预测，一共69个数据，拿后6个做测试


# %%
yre=sc.inverse_transform(yre)#反归一化

# %%
import matplotlib.pyplot as plt

# %%
yre[2583:-look_back]

# %%
len(yre)

# %%
len(origin)

# %%
from matplotlib.pyplot import MultipleLocator
plt.figure(dpi=200,figsize=(40,4))
plt.plot(origin[:-look_back],'r--',label='全部数据')
plt.plot(yre[:2583],'b-',label='验证')
plt.plot([2583+i for i in range(len(yre[2583:]))],yre[2583:],'g-',label='预测')

x_major_locator=MultipleLocator(5)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(5)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
plt.legend()

print(yre[:])

# f = open('output.txt','w')

# print(yre[2583:], file=f)
import pandas as pd

df_Mode_ = pd.DataFrame(yre[2583:])
df_Mode_.to_excel('lstm_pre_res.xlsx')

# f.close()


# %%
from sklearn.metrics import r2_score#拟合优度
print(r2_score(origin[2583:-look_back],yre[2583:]))

# %%
from sklearn.metrics import mean_squared_error,mean_absolute_error

# %%
print(mean_absolute_error(origin[2583:-look_back],yre[2583:]))#mae
print(mean_squared_error(origin[2583:-look_back],yre[2583:]))#mse

# %%
print('rmse:',mean_squared_error(origin[2583:-look_back],yre[2583:])**0.5)#rmse


