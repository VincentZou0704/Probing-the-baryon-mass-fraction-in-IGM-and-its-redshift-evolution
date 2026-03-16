import matplotlib.pyplot as plt
import emcee
import corner
import numpy as np
import get_asample
from frb_mcmc.initialization import *
import pandas as pd
import xlwt


# import samples

def getsamples(i):
    df = pd.read_excel(r'F:\pythonProject1\data_save\a0_3p_300samples.xlsx', usecols=[i])
    df_li = np.array(df.values.tolist()).flatten()
    return df_li


z = getsamples(0)
dm_frb = getsamples(1)


# nsamples = 100
# z = []
# dm_frb = []
# for i in range(nsamples):
#     z0 = get_asample.get_samples_z()
#     dm_frb0 = get_asample.get_samples_frb(z0)
#     z.append(z0)
#     dm_frb.append(dm_frb0)
# z = np.array(z).flatten()
# dm_frb = np.array(dm_frb).flatten()


def log_likelihood(theta, dm_frb, z):
    F, f, sigma_host, emu = theta
    sca = (dm_frb*(1+z)) / (scale-1)
    res = sca * (likelihood_all(devide(dm_frb*(1+z)-ie, scale),
                                sigma_host, emu, dm_frb, z, f, alpha0, F).sum(axis=0))     #integrate (sum to speed up)
    res = np.array(res) + ie  # to avoid bad things
    return np.sum(np.log(res))

#emcee
def log_prior(theta):
    F, f, sigma_host, emu = theta
    if 0.2 < sigma_host < 2 and 20 < emu < 200 and 0 < f < 1 and 0.01 < F < 0.5:
        return 0.0
    return -np.inf


def log_probability(theta, dm_frb, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dm_frb, z)


print(log_probability((0.2,0.6,1,100),dm_frb,z))
truths = [F0, f_IGM_p, sigma_host0, emu0]
pos = truths + 1e-4 * np.random.randn(20, 4)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(dm_frb, z)
)

sampler.run_mcmc(pos, 1000, progress=True)

flat_samples = sampler.get_chain(discard=100, thin=10, flat=True)

# print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
# print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))

# save mcmc results
def save_mcmc_result(flat_samples):
    my_workbook = xlwt.Workbook()
    sheet = my_workbook.add_sheet('mcmc_result')
    for i in range(flat_samples.shape[0]):
        for j in range(flat_samples.shape[1]):
            sheet.write(i, j, flat_samples[i][j])
    my_workbook.save(r'F:\pythonProject1\process\mcmc\mcmc_a0_4p+f_300.xlsx')


save_mcmc_result(flat_samples)

#plot corner
labels=[r'$F$', r'$f_{IGM,0}$','$σ_{host}$', 'exp(μ)']


if __name__ == '__main__':
    plt.figure()
    fig = corner.corner(
        flat_samples, truths=truths ,labels=labels,
        quantiles=[0.1587, 0.5, 0.8413], show_titles=True, title_kwargs={"fontsize": 12}, smooth = 1, smooth1d = 1
    )
    plt.savefig(r"F:\pythonProject1\process\figures\a0_4p+f_300.png")
    plt.show()

    # # print result
    # print_result = open(r'F:\pythonProject1\data_save\mcmc_data\result.txt', mode='a', encoding='utf-8')
    #
    # for i in range(ndim):
    #     mcmc = np.percentile(flat_samples[:, i], [15.87, 50, 84.13])
    #     q = np.diff(mcmc)
    #     print(labels[i]+"={0:.2f}".format(mcmc[1]), '\t', '68%: ' , "up:{1:.2f}, down:{0:.2f}".format(q[0],q[1]),file=print_result)
    # print('\n', file=print_result)
    # print_result.close()

