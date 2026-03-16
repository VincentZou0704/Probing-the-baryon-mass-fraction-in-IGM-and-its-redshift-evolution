import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def getdata(i):
    li = pd.read_excel(r'F:\pythonProject1\data_save\200samples\3p_results_1_200samples.xlsx', usecols=[i], header=0)
    value = np.array(li.values.tolist()).flatten()
    return  value

alpha, sigma_host, emu = getdata(0), getdata(1), getdata(2)

print(np.percentile(alpha, 16), np.percentile(alpha, 50), np.percentile(alpha, 84))
print(np.percentile(sigma_host, 16), np.percentile(sigma_host, 50), np.percentile(sigma_host, 84))
print(np.percentile(emu, 16), np.percentile(emu, 50), np.percentile(emu, 84))

plt.subplots(1, 3, sharey=True)

plt.subplot(131)
plt.hist(alpha, 10, histtype = 'step', color = 'k')
plt.subplot(132)
plt.hist(sigma_host, 10, histtype='step', color='k')
plt.subplot(133)
plt.hist(emu, 10, histtype = 'step', color = 'k')
plt.show()
