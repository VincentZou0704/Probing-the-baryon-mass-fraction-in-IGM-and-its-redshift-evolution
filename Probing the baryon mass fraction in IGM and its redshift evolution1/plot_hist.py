import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getdata(i):
    li = pd.read_excel(r'F:\pythonProject1\data_save\200samples\200samples.xlsx', usecols=[i], header=0)
    value = np.array(li.values.tolist()).flatten()
    return  value


alpha, sigma_host, emu = getdata(0), getdata(1), getdata(2)

newticks = np.linspace(0,250,11)
plt.subplots(1, 3, sharey = True)
plt.subplot(131)
height1,border1,patch1 = plt.hist(alpha, 10, histtype='step', color = 'k')
plt.plot([np.percentile(alpha,50),np.percentile(alpha,50)],[0,height1[4]], color = 'red',
         linestyle = '--', label = r'$\alpha = {0:.3f}$'.format(np.percentile(alpha,50)))
plt.legend(bbox_to_anchor=(0, 1.), loc=3, borderaxespad=0, frameon = False)
plt.yticks(newticks)
plt.xlabel('α')

plt.subplot(132)
height2,border2,patch2 = plt.hist(sigma_host, 10, histtype='step', color = 'k')
plt.plot([np.percentile(sigma_host,50),np.percentile(sigma_host,50)],[0,height2[3]], color = 'red',
         linestyle = '--', label = r'$\sigma = {0:.2f}$'.format(np.percentile(sigma_host,50)))
plt.legend(bbox_to_anchor=(0, 1.), loc=3, borderaxespad=0, frameon = False)   # num1 水平位置， num4 垂直位置
plt.xlabel(r'$\sigma_{host}$')

plt.subplot(133)
height3,border3,patch3 = plt.hist(emu, 10, histtype='step', color = 'k')
plt.plot([np.percentile(emu,50),np.percentile(emu,50)],[0,height3[3]], color = 'red',
         linestyle = '--', label = r'$exp(\mu) = {0:.1f}$'.format(np.percentile(emu,50)))
plt.legend(bbox_to_anchor=(0, 1.), loc=3, borderaxespad=0, frameon = False)
plt.xlabel(r'exp(μ)')

plt.show()

