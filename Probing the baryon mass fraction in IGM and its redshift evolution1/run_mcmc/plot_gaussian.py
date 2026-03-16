import matplotlib.pyplot as plt
import emcee
import xlwt
import get_samples_inverseF
from frb_mcmc.initialization import *


times = 100

def log_likelihood(theta, dm_frb, z):
    alpha, sigma_host, emu = theta
    sca = (dm_frb*(1+z)) / (scale-1)
    res = sca * (likelihood_all(devide(dm_frb*(1+z)-ie, scale),
                                sigma_host, emu, dm_frb, z, f_IGM_p, alpha, F0).sum(axis=0))     #integrate (sum to speed up)
    res = np.abs(res) + ie  # to avoid bad things
    return np.sum(np.log(res))

#emcee
def log_prior(theta):
    alpha, sigma_host, emu = theta
    if 0.2 < sigma_host < 2 and 20 < emu < 200 and -2 < alpha < 2:
        return 0.0
    return -np.inf


def log_probability(theta, dm_frb, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dm_frb, z)

pos = [alpha0, sigma_host0, emu0] + 1e-4 * np.random.randn(6, 3)

nwalkers, ndim = pos.shape

myworkbook = xlwt.Workbook()
sheet = myworkbook.add_sheet('sheet')
sheet.write(0,0,'alpha')
sheet.write(0,1,'sigma_host')
sheet.write(0,2,'emu')

for i in range(times):
    z = get_samples_inverseF.get_samples_z(3)
    dm_frb = get_samples_inverseF.get_samples_frb(z)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(dm_frb, z)
    )

    sampler.run_mcmc(pos, 500, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=20, flat=True)

    alpha, sigma_host, emu = np.percentile(flat_samples[:,0],50), \
                 np.percentile(flat_samples[:,1],50), \
                 np.percentile(flat_samples[:,2],50)
    sheet.write(i + 1, 0, alpha)
    sheet.write(i + 1, 1, sigma_host)
    sheet.write(i + 1, 2, emu)
    print('process:', i+1, '%', 'result:', alpha, sigma_host, emu)

if __name__ == '__main__':
    myworkbook.save(r'F:\pythonProject1\data_save\3p_a0_100_distribution_save.xlsx')

