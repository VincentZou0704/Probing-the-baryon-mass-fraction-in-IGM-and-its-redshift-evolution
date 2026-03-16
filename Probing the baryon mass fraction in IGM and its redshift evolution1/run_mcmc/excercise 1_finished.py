import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize
import emcee
import corner
from FRBpopulation import FuncZou as fz



def excel_one_line_to_list(i):
    df = pd.read_excel(r"C:\Users\xinjing\Desktop\工作区\暑假任务\暑假任务\Union2.1.xls", usecols=[i], names=None)
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    return result


if __name__ == '__main__':
    z = excel_one_line_to_list(0)
if __name__ == '__main__':
    mu = excel_one_line_to_list(1)
if __name__ == '__main__':
    delta_mu = excel_one_line_to_list(2)

#编写需拟合的函数
c = 3 * 10 ** 5
def integ(x, omega):
    return (omega * (1 + x) ** 3 + (1 - omega)) ** (-1 / 2)


plt.hist(np.array(mu), bins = 'auto')
plt.show()


#
# def log_likelihood(theta, x, y, yerr):
#     res = []
#     x = np.array(x)
#     y = np.array(y)
#     yerr = np.array(yerr)
#     h0, omega = theta
#     for i in range(len(x)):
#         i0, i1 = integrate.quad(integ, 0, x[i], args=omega)
#         res.append(i0)
#
#     model = 5 * np.log10((1 + x) * c / h0 * res) + 25
#     sigma2 = yerr ** 2
#     return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
#
# print(log_likelihood([67, 0.3], [0.02], [35], [0.23]))    #检验函数正确性，结果正确
#
# #拟合最大似然概率的参数，暂时不用
#
# np.random.seed(42)
# nll = lambda *args: -log_likelihood(*args)
# initial = np.array([67, 0.3]) + 0.1 * np.random.randn(2)
# soln = minimize(nll, initial, args=(z, mu, delta_mu))
# h0_ml, omega_ml = soln.x
# print(h0_ml,omega_ml)
#
#
# def log_prior(theta):
#     h0, omega = theta
#     if 0 < h0 < 100 and 0 < omega < 1:
#         return 0.0
#     return -np.inf
#
#
# def log_probability(theta, x, y, yerr):
#     lp = log_prior(theta)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + log_likelihood(theta, x, y, yerr)
#
#
# pos = soln.x + 1e-4 * np.random.randn(25, 2)   #后验值猜测
# nwalkers, ndim = pos.shape
#
# sampler = emcee.EnsembleSampler(
#     nwalkers, ndim, log_probability, args=(z, mu, delta_mu)
# )
#
# sampler.run_mcmc(pos, 3000, progress=True)
#
# flat_samples = sampler.get_chain(discard=100, thin=10, flat=True)
# print(flat_samples.shape)
#
# labels=['h0','Ωm']
#
# plt.figure()
# fig = corner.corner(
#     flat_samples, labels=labels, truths=[h0_ml,omega_ml]
# )
# plt.savefig("excercise_1")
# plt.show()
#
# for i in range(ndim):
#     mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
#     q = np.diff(mcmc)
#     print(labels[i] + "={0:.3f}-{1:.3f}+{2:.3f}".format(mcmc[1], q[0], q[1]))
#
#
