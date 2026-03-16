import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import corner
import xlwt
from frb_mcmc import splinedata as sd

dmfrb = np.array([361.42, 589.27, 363.6, 338.7, 593.1])  #此项中仅包含DM_host和DM_cosmic, 338.7, 593.1
dmism = np.array([41.89419211467203, 42.39581689047655, 56.815577390267244, 38.063401552135424, 55.93063340127531])  #从数据库中得出 , 38.063401552135424, 55.93063340127531
dmhalo = np.array([50]*5)
dm0 = dmfrb - dmism - dmhalo
print(dm0)
z0 = np.array([0.3214, 0.4755, 0.291, 0.1178, 0.522])  # ,0.1178,0.522
h0_P = 67.74
ob_P = 0.0486
f_IGM_p = 0.83
a2 = 0.691
a1 = 1 - a2
#c0 = np.array([1.9239, 1.7477, 1.9700, 2.3714, 1.7067])  #固定F=0.2解出的与红移一一对应的C0值, 2.3713713713713718, 1.7067067067067068
#a = np.array([0.3802, -1.9527, 2.5970, 1.5797])   #多项式拟合的C0关于sigma_2的参数
scale = 1000
coefficient = 21*3*10**5/64/np.pi/6.67/1.67/3.085677581467**2    # 1MPC = 3.0857*10**22 m 还要乘H0,omega,f_IGM才是完整的系数

splc = sd.splinec0()
spla = sd.splineA()
splz = sd.splinehez(a1)


def devide(n, nscale):
    ret = np.linspace(1 * 10 ** (-5), n, nscale)     #return a shape of (nscale, len(n))
    return ret


def dm_c_a(ob_h, z):   #dm_cosmic_average,h0 hided in omega_b
    return coefficient*f_IGM_p*splz(z)*ob_h*70*(1+0.25*z/(1+z))
print(dm_c_a(ob_P*h0_P/70, z0))


def like_host(dm_h, s_h, emu): #已检验,函数正确
    return  np.exp(-0.5*(np.log(2*np.pi)+2*np.log(dm_h*s_h)+((np.log(dm_h/emu))**2)/s_h**2))


def like_cos(dm_h, dm, z, ob_h, F): #已检验,函数正确
    dm_c = dm - dm_h/(1+z)
    delta = dm_c/dm_c_a(ob_h, z)
    sigma = F/np.sqrt(z)
    c = splc(sigma)
    A = spla(sigma)
    x = (delta**(-3)-c)**2/18/sigma**2
    return  A*delta**(-3)*np.exp(- x)       #c值对结果影响不大


def combi(dm_h, s_h, emu, dm, z, ob_h, F): #已检验,函数正确
    return like_host(dm_h, s_h, emu)*like_cos(dm_h, dm, z, ob_h, F)



def log_likelihood(theta, dm, z):
    F, ob_h, s_h, emu = theta
    sca = (dm*(1+z)) / (scale-1)
    res = sca * (combi(devide(dm*(1+z)-1 * 10 ** (-5), scale), s_h, emu, dm, z, ob_h, F).sum(axis=0))  #test in integrate_test
    res = np.array(res) + 1 * 10 ** (-5)
    return np.sum(np.log(res))

#从此处开始不会出现问题
def log_prior(theta):
    F, ob_h, s_h, emu = theta
    if 0.2 < s_h < 2 and 20 < emu < 200 and 0.015 < ob_h < 0.095 and 0.011 < F < 0.5:
        return 0.0
    return -np.inf


def log_probability(theta, dm, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dm, z)



pos = [0.3, 0.056, 0.88, 68.2] + 1e-4 * np.random.randn(25, 4)   #后验值猜测(取值无关紧要)

nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(dm0, z0)
)

sampler.run_mcmc(pos, 10000, progress=True)

flat_samples = sampler.get_chain(discard=200, thin=20, flat=True)
print(flat_samples.shape)

labels = ['F', '$Ω_b h_{70}$', '$σ_{host}$', 'exp(μ)']
#将取样数据保存
# 新建工作簿
my_workbook = xlwt.Workbook()
# 创建新的工作表对象 worksheet, 并取名为 test_sheet
sheet = my_workbook.add_sheet('test_sheet')
# 给定单元格坐标, 一次写入一个单元格数据
sheet.write(0, 0, "F")
sheet.write(0, 1, "Ω_bh_70") #sheet.write(row行,col列,value)#
sheet.write(0, 2, "σ_host")
sheet.write(0, 3, "exp(μ)")
# 保存文件
print(flat_samples.shape[0])
for i in range(flat_samples.shape[0]):
    for j in range(0, len(labels)):
        sheet.write(i+1, j, flat_samples[i][j])
my_workbook.save('nature_1(5).xlsx')

plt.figure()
fig = corner.corner(
    flat_samples, labels=['F', '$Ω_b h_{70}$', '$σ_{host}$', 'exp(μ)',"$\Gamma \, [\mathrm{parsec}]$"],
    quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs={"fontsize": 12}, smooth = 1, smooth1d= 3
)
plt.savefig("nature_1(5)")
plt.show()

prinres = open('prinres.txt', mode='a', encoding='utf-8')

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
