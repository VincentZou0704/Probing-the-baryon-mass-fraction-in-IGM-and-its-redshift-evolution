import numpy as np
import pandas as pd
from frb_mcmc.initialization import *


def getsamples(i):
    df = pd.read_excel(r'F:\pythonProject1\data_save\mcmc_data\mcmc_a0_4p_300.xlsx', usecols=[i], header=None)
    df_li = np.array(df.values.tolist()).flatten()
    return df_li

i=0

while True:
    mcmc = np.percentile(getsamples(i), [16, 50, 84])
    q = np.diff(mcmc)
    print('mid:{0:.2f}, +{2:.2f}, -{1:.2f}'.format(np.percentile(getsamples(i),50), q[0], q[1]))
    i = i+1

