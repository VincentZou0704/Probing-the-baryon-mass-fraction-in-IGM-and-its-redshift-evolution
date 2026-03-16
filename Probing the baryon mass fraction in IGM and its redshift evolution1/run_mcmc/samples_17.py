import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from frb_mcmc import splinedata as sd
import xlwt
from scipy.integrate import quad


dmfrb = np.array([557.00, 536.00, 348.80, 362.16, 103.50, 589.00,
                  364.55, 760.80, 340.05, 332.63, 592.60, 504.13,
                  507.90, 297.50, 380.25, 577.80, 413.52])  # delete 181030[4] and 190611[9]
dmism = np.array([157.60, 136.53, 168.73, 41.45, 40.16, 41.98,
                  56.22, 36.74, 37.81, 56.60, 55.37, 38.00,
                  44.22, 33.75, 27.35, 36.19, 126.49])  # 从数据库中得出

dmhalo = np.array([50]*17)
dm0 = dmfrb - dmism - dmhalo    # 此项中仅包含DM_host和DM_cosmic
dm0 = list(dm0) + [114 - 37] # FRB 20171020A and rFRB 20190520B DM_E[114 - 37, 1205 - 60]
dm0 = np.array(dm0)
ie = 1e-11

z0 = [0.1927,0.3305,0.0337,0.3214,0.0039,0.4755,
    0.2913,0.6600,0.1178,0.3778,0.5217,0.2365,
    0.2340,0.2432,0.1608,0.3688,0.0979] + [0.0087]   # z [0.0087, 0.241]


z0 = np.array(z0)
print('z_list:',z0, '\n', 'dm_e_list:', dm0)

h0_P = 67.4
ob_P = 0.0493
f_IGM_p = 0.84
a1 = 0.315
a2 = 1 - a1
coefficient = 21*2.99792458*10**5/64/np.pi/6.67408/1.672621637/3.085677581467**2
scale = 1000

splc = sd.splinec0()
spla = sd.splineA()
splz = sd.splinehez(a1)

labels = [r'$\alpha$', r'$σ_{host}$', 'exp(μ)']

# r'$F$', r'$f_{IGM,0}$', r'$\alpha$', r'$σ_{host}$', 'exp(μ)'

def devide(n, nscale):
    ret = np.linspace(ie, n, nscale)     #return a shape of (nscale, len(n))
    return ret

def dm_c_a(f, alpha, z):    #   f parameterized as f = f_IGM,0 * (1 + alpha * z/(1+z) )
    return coefficient*splz(z)*ob_P*h0_P*f*(1+alpha*z/(1+z))

def like_host(dm_h, s_h, emu):
    return  np.exp(-0.5*(np.log(2*np.pi)+2*np.log(dm_h*s_h)+((np.log(dm_h/emu))**2)/s_h**2))


def like_cos(dm_h, dm, z, f, alpha, F):
    dm_c = dm - dm_h/(1+z)
    delta = dm_c/dm_c_a(f, alpha, z)
    sigma = F/np.sqrt(z)       #you can fix F = 0.2
    c = splc(sigma)
    A = spla(sigma)
    x = (delta**(-3)-c)**2/18/sigma**2
    return  A*delta**(-3)*np.exp(- x)


def combi(dm_h, s_h, emu, dm, z, f, alpha, F): #已检验,函数正确
    return like_host(dm_h, s_h, emu)*like_cos(dm_h, dm, z, f, alpha, F)


# def log_likelihood(theta, dm, z):
#     alpha, s_h, emu = theta
#     # F, f_IGM, alpha, s_h, emu
#     sca = (dm*(1+z)) / (scale-1)
#     res = sca * (combi(devide(dm*(1+z) - ie, scale), s_h, emu, dm, z, f_IGM_p, alpha, 0.2).sum(axis=0))     #integrate (sum to speed up)
#     res = np.array(res) + ie  # to avoid bad things
#     return np.sum(np.log(res))

def log_likelihood(theta, dm, z):
    res = []
    alpha, s_h, emu = theta
    for i in range(len(dm)):
        i0, i1 = quad(combi, 0, dm[i]*(1+z[i]), args=(s_h, emu, dm[i],z[i],f_IGM_p, alpha, 0.2))
        res.append(i0)
    return np.sum(np.log(res))



def log_prior(theta):
    alpha, s_h, emu = theta
    if 0.2 < s_h < 2 and 20 < emu < 200  and -2 < alpha < 2:
        # 0.2 < s_h < 2 and 20 < emu < 200  and -2 < alpha < 2 and 0.01< F < 0.5 and 0 < f_IGM < 1
        return 0.0
    return -np.inf


def log_probability(theta, dm, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dm, z)


pos = [0.2, 1, 100] + 1e-4 * np.random.randn(10, len(labels))   #后验值猜测(取值无关紧要)
# 0.2, 0.84, 0.2, 1, 100
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(dm0, z0)
)

sampler.run_mcmc(pos, 1000, progress=True)

flat_samples = sampler.get_chain(discard=200, thin=20, flat=True)


if __name__ == '__main__':
    plt.figure()
    fig = corner.corner(
        flat_samples, labels = labels, show_titles = True,
        quantiles=[0.16, 0.5, 0.84], smooth=1, smooth1d=3
    )
    # plt.savefig(r"F:\pythonProject1\process\figures\18samples_5p.png")
    plt.show()

    #
    # my_workbook = xlwt.Workbook()
    # sheet = my_workbook.add_sheet('mcmc_result')
    # for i in range(flat_samples.shape[0]):
    #     for j in range(flat_samples.shape[1]):
    #         sheet.write(i, j, flat_samples[i][j])
    # my_workbook.save(r'F:\pythonProject1\process\mcmc\18samples_5p.xlsx')
    #
    #
    # # print result
    #
    # print_result = open(r'F:\pythonProject1\process\19samples.txt', mode='a', encoding='utf-8')
    #
    # for i in range(ndim):
    #     mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    #     q = np.diff(mcmc)
    #     print(labels[i]+"={0:.3f}".format(mcmc[1]), '\t', '68%: ' , "-:{0:.3f}, +:{1:.3f}".format(q[0],q[1]),
    #           file=print_result)
    # print('\n', file=print_result)
    # print_result.close()
