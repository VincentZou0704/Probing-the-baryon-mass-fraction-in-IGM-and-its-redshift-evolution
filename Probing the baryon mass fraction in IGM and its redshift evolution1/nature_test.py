import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import corner
import xlwt
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


dmfrb = np.array([361.42, 589.27, 363.6, 338.7, 593.1])  #此项中仅包含DM_host和DM_cosmic, 338.7, 593.1
dmism = np.array([41.89419211467203, 42.39581689047655, 56.815577390267244, 38.063401552135424, 55.93063340127531])  #从数据库中得出 , 38.063401552135424, 55.93063340127531
dmhalo = np.array([50]*5)
dm0 = dmfrb - dmism - dmhalo
print(dm0)
z0 = np.array([0.3214, 0.4755, 0.291, 0.1178, 0.522])  # ,0.1178,0.522
h0_P = 67.74
ob_P = 0.0486
f_IGM_p = 0.84
a2 = 0.691
a1 = 1 - a2
#c0 = np.array([1.9239, 1.7477, 1.9700, 2.3714, 1.7067])  #固定F=0.2解出的与红移一一对应的C0值, 2.3713713713713718, 1.7067067067067068
#a = np.array([0.3802, -1.9527, 2.5970, 1.5797])   #多项式拟合的C0关于sigma_2的参数
scale = 1000
coefficient = 21*3*10**5/64/np.pi/6.67/1.67/3.085677581467**2    # 1MPC = 3.0857*10**22 m 还要乘H0,omega,f_IGM才是完整的系数


#读取sigma_co spline
def excel_one_line_to_list_c(i):
    df = pd.read_excel(r"F:\pythonProject1\sigma_c0.xlsx", usecols=[i], names=None)
    # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    return result


if __name__ == '__main__':
    sigma0 = excel_one_line_to_list_c(0)
if __name__ == '__main__':
    c0 = excel_one_line_to_list_c(1)


splc = IUS(sigma0, c0)

#读取sigma_A spline
def excel_one_line_to_list_a(i):
    df = pd.read_excel(r"F:\pythonProject1\spline_A.xlsx", usecols=[i], names=None)
    # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    return result


if __name__ == '__main__':
    sigma1 = excel_one_line_to_list_a(0)
if __name__ == '__main__':
    A = excel_one_line_to_list_a(1)

spla = IUS(sigma1, A)


def devide(n, nscale):
    ret = np.linspace(1 * 10 ** (-5), n, nscale)     #return a shape of (nscale, len(n))
    return ret


def dm_c_a(ob_h, f_IGM, z):   #dm_cosmic_average,h0 hided in omega_b
    sca = z/(scale-1)
    x = devide(z,scale)
    f = sca*(1+x)/np.sqrt(a1*(1+x)**3+a2)
    f_sum = f.sum(axis = 0)
    return coefficient*f_IGM*f_sum*ob_h*70
print(dm_c_a(ob_P*h0_P/70, f_IGM_p, z0))


def like_host(dm_h, s_h, emu): #已检验,函数正确
    return  np.exp(-0.5*(np.log(2*np.pi)+2*np.log(dm_h*s_h)+((np.log(dm_h/emu))**2)/s_h**2))


def like_cos(dm_h, dm, z, ob_h, f_IGM, F): #已检验,函数正确
    dm_c = dm - dm_h/(1+z)
    delta = dm_c/dm_c_a(ob_h, f_IGM, z)
    sigma = F/np.sqrt(z)
    c = splc(sigma)
    A = spla(sigma)
    x = (delta**(-3)-c)**2/18/sigma**2
    return  A*delta**(-3)*np.exp(- x)       #c值对结果影响不大


def combi(dm_h, s_h, emu, dm, z, ob_h, f_IGM, F): #已检验,函数正确
    return like_host(dm_h, s_h, emu)*like_cos(dm_h, dm, z, ob_h, f_IGM, F)



def log_likelihood(theta, dm, z):
    F, f_IGM, ob_h, s_h, emu = theta
    sca = dm / (scale-1)
    res = sca * (combi(devide(dm*(1+z)-1 * 10 ** (-5), scale), s_h, emu, dm, z, ob_h, f_IGM, F).sum(axis=0))  #test in integrate_test
    res = np.array(res) + 1 * 10 ** (-5)
    return np.sum(np.log(res))

#从此处开始不会出现问题
def log_prior(theta):
    F, f_IGM, ob_h, s_h, emu = theta
    if 0.2 < s_h < 2 and 20 < emu < 200 and 0.015 < ob_h < 0.095 and 0.011 < F < 0.5 and 0 < f_IGM < 1:
        return 0.0
    return -np.inf


def log_probability(theta, dm, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dm, z)



pos = [0.3, 0.84, 0.056, 0.88, 68.2] + 1e-4 * np.random.randn(25, 5)   #后验值猜测(取值无关紧要)

nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(dm0, z0)
)

sampler.run_mcmc(pos, 10000, progress=True)

flat_samples = sampler.get_chain(discard=200, thin=20, flat=True)
print(flat_samples.shape)

labels = ['F', '$f_{IGM}$','$Ω_b h_{70}$', '$σ_{host}$', 'exp(μ)']

plt.figure()
fig = corner.corner(
    flat_samples, labels=['F', '$f_{IGM}$', '$Ω_b h_{70}$', '$σ_{host}$', 'exp(μ)',"$\Gamma \, [\mathrm{parsec}]$"],
    quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs={"fontsize": 12}, smooth = 1, smooth1d= 3
)
plt.savefig("nature_test")
plt.show()

prinres = open('run_mcmc/prinres.txt', mode='a', encoding='utf-8')

print("nature 68%置信结果如下：",file=prinres)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(labels[i] + "={0:.3f}下限{1:.3f}上限{2:.3f}".format(mcmc[1],mcmc[1]-q[0],mcmc[1]+q[1]),file=prinres)

print("nature 95%置信结果如下：",file=prinres)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [2.5, 50, 97.5])
    q = np.diff(mcmc)
    print(labels[i] + "={0:.3f}下限{1:.3f}上限{2:.3f}".format(mcmc[1],mcmc[1]-q[0],mcmc[1]+q[1]),file=prinres)
print('\n')
prinres.close() # 关闭文件
