import xlwt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


def getdata(i):
    li = pd.read_excel(r'F:\pythonProject1\data_save\200samples\3p_results_1_200samples.xlsx', usecols=[i], header=0)
    values = np.array(li.values.tolist()).flatten()
    return values


def devide(_min_, _max_, nscale):
    ret = np.linspace(_min_, _max_, nscale)     # return a shape of (nscale, 1)
    return ret


def CDF_samples(min, max, x):
    intg = ((max - min)/(scale - 1)) * spl_data(devide(min,max,scale)).sum(axis=0)
    normalization = 1/intg
    piece = (x - min)/(scale-1)
    cdf = piece * normalization * spl_data(devide(min,x,scale)).sum(axis=0)
    return cdf


alpha, sigma_host, emu = getdata(0), getdata(1), getdata(2)
nbins = 5

real, i = alpha, 0
plt.subplot(121)
height, border, patchs = plt.hist(real, nbins, histtype = 'step', color = 'k')
print( 'border:',border, '\n','height:', height)
height_y = np.array([height[0]/2] + list(height) + [height[nbins-1]/2])
border_diff = np.diff(border)
# If list a single number, [] is essential and enough, if a set of number, should state 'list' and omit []
border_x = np.array([border[0]] + list(border[0:nbins] + border_diff/2)+ [border[nbins]])

print('border_x:',border_x, '\n','height_y:', height_y)
spl_data = IUS(border_x,height_y)
scale, nsamples = 1000, 1000
min, max = np.min(border_x), np.max(border_x)
x = devide(min,max,scale)
plt.plot(x,spl_data(x))

spl_getsamples = IUS(CDF_samples(min,max,x),x)
U_x = np.random.rand(nsamples)
results = spl_getsamples(U_x)
print('real:',np.mean(real),'\t','model:',np.mean(results))

labels = ['α', 'σ_host', 'emu']

mcmc = np.percentile(results, [16, 50, 84])
print(labels[i] + "={0:.3f}".format(mcmc[1]), '\t', '68%: ', "down:{0:.3f}, up:{1:.3f}".format(mcmc[0], mcmc[2]))

plt.subplot(122)
plt.hist(results,20,histtype = 'step',color = 'k')

plt.show()


myworkbook = xlwt.Workbook()
sheet = myworkbook.add_sheet('sheet')
for i in range(nsamples):
    sheet.write(i,0,results[i])
myworkbook.save(r'F:\pythonProject1\frb_mcmc\method_test\test_result.xlsx')
