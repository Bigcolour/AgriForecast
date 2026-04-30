import numpy as np
import matplotlib.pyplot as plt
import ewtpy
import pandas as pd
import datetime

df = pd.read_excel('2_分解/data_3228.xlsx')
f = df['Price'].values
T = len(f)
fs = 1/T
t = np.arange(1, T+1)/T

N = 7 #number of supports
detect = "locmax" #detection mode: locmax, locmaxmin, locmaxminf
reg = 'none' #spectrum regularization - it is smoothed with an average (or gaussian) filter 
lengthFilter = 0 #length or average or gaussian filter
sigmaFilter = 0 #sigma of gaussian filter
Fs = 1 #sampling frequency, in Hz (if unknown, set 1)

ewt,  mfb ,boundaries = ewtpy.EWT1D(f, 
                                    N = N,
                                    log = 0,
                                    detect = detect, 
                                    completion = 0, 
                                    reg = reg, 
                                    lengthFilter = lengthFilter,
                                    sigmaFilter = sigmaFilter)

#plot original signal and decomposed modes

# plt.title('EWT modes')
print(ewt,'ewt')
_ewt = ewt.swapaxes(0, 1)

fig, axs = plt.subplots(N+1, 1, figsize=(8, (N+1)*1.3), sharex=True)
# axs[0].set_title('EWT modes')
for i in range(N):
    axs[i].plot(t, _ewt[i])
    axs[i].set_ylabel(f'Mode {i+1}')
    
# Calculate and plot residuals
residuals = f - np.sum(_ewt, axis=0)
axs[N].plot(t, residuals, color='red')
axs[N].set_ylabel('Residual')
# plt.xlabel('Samples')
# fig.suptitle('Empirical Wavelet Transform Modes', fontsize=16)

# #%% show boundaries
# ff = np.fft.fft(f)
# freq=2*np.pi*np.arange(0,len(ff))/len(ff)

# if Fs !=-1:
#     freq=freq*Fs/(2*np.pi)
#     boundariesPLT=boundaries*Fs/(2*np.pi)
# else:
#     boundariesPLT = boundaries

# ff = abs(ff[:ff.size//2])#one-sided magnitude
# freq = freq[:freq.size//2]

# plt.figure()
# plt.plot(freq,ff)
# for bb in boundariesPLT:
#     plt.plot([bb,bb],[0,max(ff)],'r--')
# plt.title('Spectrum partitioning')
# plt.xlabel('Hz')
plt.show()

# Save results to a single Excel file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
Mode_data = {'Mode{}'.format(i+1):  _ewt[i] for i in range(N)}
Mode_data['Residual'] = f - np.sum( _ewt, axis=0)
df_Mode = pd.DataFrame(Mode_data)
df_Mode.to_excel(f'2_分解/EWT/结果/ewt_result_{timestamp}.xlsx')