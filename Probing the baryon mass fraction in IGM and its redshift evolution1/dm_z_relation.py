import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlwt
from frb_mcmc import splinedata
from frb_mcmc.initialization import *

def getsamples(i):
    df = pd.read_excel(r'F:\pythonProject1\frb_mcmc\cosmic_sigma.xlsx', usecols=[i] )
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    return result


dm_frb_repeat = np.array([557.00, 536.00, 348.80, 103.50, 592.60,413.52])
dm_frb_nonrepeat = np.array([ 362.16, 589.00, 364.55, 760.80, 340.05, 332.63,
                              504.13, 507.90, 297.50, 380.25, 577.80])
dm_ism_repeat = np.array([158.23, 133.01, 171.67, 40.16,55.93,123.16])
dm_ism_nonrepeat = np.array([41.89,42.39,  56.81, 36.79, 38.06, 57.19,
                             37.98,44.70, 33.88, 27.01, 35.92])
z_repeat = np.array([0.1927,0.3305,0.0337,0.0039,0.5217,0.0979])
z_nonrepeat = np.array([0.3214,0.4755,0.2913,0.6600,0.1178,0.3778
                       ,0.2365,0.2340,0.2432,0.1608,0.3688])
dmhalo1 = np.array([50]*6)
dmhalo2 = np.array([50]*11)
dm0_repeat = dm_frb_repeat - dmhalo1 - dm_ism_repeat
dm0_nonrepeat = dm_frb_nonrepeat - dmhalo2 - dm_ism_nonrepeat
dm_cosmic_repeat = dm0_repeat - 50/(1+z_repeat)
print(dm_cosmic_repeat)
dm_cosmic_nonrepeat = dm0_nonrepeat - 50/(1+z_nonrepeat)


sigma_down = getsamples(1)
sigma_up = getsamples(0)
z = np.linspace(0.001,0.7,1000)
splhez = splinedata.splinehez(0.315)
_up = dm_cosmic_average(z,0.84,0.2)*sigma_up
_down = dm_cosmic_average(z,0.84,0.2)*sigma_down



plt.plot(z,dm_cosmic_average(z,0.84,0.2),'k',linewidth = 1,label = '<$\mathrm{DM_{IGM}}(z)$>')
plt.fill_between(z,_up,_down,facecolor = '#f9d3e3',edgecolor = 'white',alpha = 0.3)
plt.plot(z_repeat,dm_cosmic_repeat,'s',color = '#50c878', linewidth = 1.0, label = 'repeating FRBs')
plt.plot(z_nonrepeat,dm_cosmic_nonrepeat,'o',color = '#fbd26a', linewidth = 1.0,label = 'non-repeating FRBs')
plt.xlabel(r'$z$')
plt.ylabel(r'$\mathrm{DM_{IGM}}(pc cm^{-3})$')
plt.xlim((0,0.7))
plt.ylim((-100,1000))
plt.legend(loc='upper left',edgecolor = 'white')
plt.show()