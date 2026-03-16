import numpy as np
from frb_mcmc.settings import *
import matplotlib.pyplot as plt
import emcee
import corner
import pandas as pd

dmfrb = np.array([557.00, 536.00, 348.80, 362.16, 103.50, 589.00,
                  364.55, 760.80, 340.05, 332.63, 592.60, 504.13,
                  507.90, 297.50, 380.25, 577.80, 413.52])  # delete 181030[4] and 190611[9]
dmism = np.array([157.60, 136.53, 168.73, 41.45, 40.16, 41.98,
                  56.22, 36.74, 37.81, 56.60, 55.37, 38.00,
                  44.22, 33.75, 27.35, 36.19, 126.49])  #从数据库中得出
dmhalo = np.array([50]*17)
dm0 = dmfrb - dmism - dmhalo    #此项中仅包含DM_host和DM_cosmic
#dm0 = list(dm0)
#del dm0[16], dm0[13], dm0[12], dm0[9], dm0[7], dm0[4], dm0[2], dm0[1], dm0[0]
#dm0 = np.array(dm0)
ie = 1e-11

z0 = [0.1927,0.3305,0.0337,0.3214,0.0039,0.4755,
    0.2913,0.6600,0.1178,0.3778,0.5217,0.2365,
    0.2340,0.2432,0.1608,0.3688,0.0979]

#del z0[16], z0[13], z0[12], z0[9], z0[7], z0[4], z0[2], z0[1], z0[0]

z0 = np.array(z0)

def dm_gamma(z, m_photon):
    revision = 1e-50
    k = revision**2*4*np.pi**2*m_electron*epsilon*c_photon**5/h_planck**2/q_electron**2/h0_P
    return k*splhgamma(z)*m_photon**2/1000


def devide(n, nscale):
    ret = np.linspace(ie, n, nscale)     #return a shape of (nscale, len(n))
    return ret


def dm_cosmic_average(z, f, alpha):        #   f parameterized as f = f_IGM,0 * (1 + alpha * z/(1+z) )
    #f = 0.84
    #alpha = 0
    return coefficient*ob_P*h0_P*f * splz(z)*(1+alpha*z/(1+z))


def likelihood_host(dm_host, sigma_host, emu):
    x = np.log(np.abs(dm_host/emu)+ie)**2/(2*sigma_host**2)
    return  np.exp(-x)/(np.sqrt(2*np.pi)*dm_host*sigma_host)


def likelihood_cosmic(dm_host, dm_frb, z, f, alpha, F, m_photon):
    dm_cosmic = dm_frb - dm_host/(1+z) - dm_gamma(z, m_photon)
    delta = (dm_cosmic/dm_cosmic_average(z, f , alpha) + ie)
    sigma = np.abs(F/np.sqrt(z))       #fix F = 0.2
    c = splc(sigma)
    A = spla(sigma)
    x = (delta**(-3)-c)**2/18/sigma**2
    return  A*delta**(-3)*np.exp(- x)


def likelihood_all(dm_host, sigma_host, emu, dm_frb, z, f, alpha, F, m_photon):
    return likelihood_host(dm_host, sigma_host, emu)*likelihood_cosmic(dm_host, dm_frb, z, f, alpha, F, m_photon)


def log_likelihood(theta, dm_frb, z):
    F, sigma_host, emu, m_photon= theta
    upper = (dm_frb - dm_gamma(z, m_photon))*(1+z)
    sca = upper / (scale-1)
    res = sca * (likelihood_all(devide(upper-ie, scale),
                                sigma_host, emu, dm_frb, z, f_IGM_p, alpha0, F, m_photon).sum(axis=0))     #integrate (sum to speed up)
    res = np.array(res) + ie  # to avoid bad things
    return np.sum(np.log(res))

#emcee
def log_prior(theta):
    F, sigma_host, emu, m_photon = theta
    if 0.2 < sigma_host < 2 and 20 < emu < 200 and 0 < m_photon < 10 and 0.01 < F < 0.5:
        return 0.0
    return -np.inf


def log_probability(theta, dm_frb, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dm_frb, z)


pos = [F0,sigma_host0, emu0, 0] + 1e-4 * np.random.rand(20, 4)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(dm0, z0)
)

sampler.run_mcmc(pos, 500, progress=True)

flat_samples = sampler.get_chain(discard=100, flat=True)

# print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
# print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))



#plot corner
labels = ['σ_host', 'exp(μ)', 'm_γ']


if __name__ == '__main__':

    plt.figure()
    fig = corner.corner(
        flat_samples, labels=[r'$F$','$σ_{host}$', 'exp(μ)', r'$m_\gamma$'],
        show_titles=True,title_kwargs={"fontsize": 12},smooth=1, smooth1d=3
    )
    plt.savefig(r"F:\pythonProject1\figure\photon_3p.png")
    plt.show()
    print('m_gamma < :', np.percentile(flat_samples[:,3],68), 'e-50')


