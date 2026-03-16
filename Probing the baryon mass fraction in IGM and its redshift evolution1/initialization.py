from frb_mcmc.settings import *
import matplotlib.pyplot as plt


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


def likelihood_cosmic(dm_host, dm_frb, z, f, alpha, F):
    dm_cosmic = dm_frb - dm_host/(1+z)
    delta = (dm_cosmic/dm_cosmic_average(z, f , alpha) + ie)
    sigma = np.abs(F/np.sqrt(z))       #fix F = 0.2
    c = splc(sigma)
    A = spla(sigma)
    x = (delta**(-3)-c)**2/18/sigma**2
    return  A*delta**(-3)*np.exp(- x)


def likelihood_all(dm_host, sigma_host, emu, dm_frb, z, f, alpha, F):
    return likelihood_host(dm_host, sigma_host, emu)*likelihood_cosmic(dm_host, dm_frb, z, f, alpha, F)


if __name__ == '__main__':
    dm_host = np.linspace(ie,400,1000)
    plt.plot(dm_host, likelihood_host(dm_host, 1, 100))
    plt.show()