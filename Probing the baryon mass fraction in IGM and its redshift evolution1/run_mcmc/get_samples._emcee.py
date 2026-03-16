import emcee
import xlwt
from frb_mcmc.settings import *

nsamples = 100  # Must be a multiple of 20

discard = 100
thin = 20

spldc = sd.splinedcz(a1)

def dm_cosmic_average(z, f, alpha):    #   f parameterized as f = f_IGM,0 * (1 + alpha * z/(1+z) )
    #f = 0.84
    #alpha = 0.2
    return coefficient*splz(z)*ob_P*h0_P*f*(1+alpha*z/(1+z))

# get samples of z

def log_likelihood_z(z):
    a, b, c, B, C, eta = 3.4, -0.3, -3.5, 5000, 9, -10
    SFR_z = 0.02*((1+z)**(a*eta)+((1+z)/B)**(b*eta)+((1+z)/C)**(c*eta))**(1/eta)
    H_z = np.sqrt(a1*(1+z)**3+a2)
    return np.log(spldc(z)**2*SFR_z/(1+z)/H_z)          #  emcee actually requires the logarithm of P

def prior_z(z):
    if 0.01 < z < 3:
        return 0.0
    return -np.inf

def log_probability_z(z):
    lp = prior_z(z)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_z(z)
pos_z = [1.72] + 1e-4 * np.random.randn(20, 1)

nwalkers, ndim = pos_z.shape

sampler_z = emcee.EnsembleSampler(nwalkers, ndim, log_probability_z)

sampler_z.run_mcmc(pos_z, nsamples + discard, progress=True)

samples_z = sampler_z.get_chain(discard = discard, thin = thin, flat=True)       # This call can return z with shape(100,1)

print('z mean:{0:.3f}, min:{1:.3f}, max:{2:.3f}'.
      format(np.mean(samples_z), np.min(samples_z), np.max(samples_z)))     # mean = 1.7217

cosmic_average = dm_cosmic_average(samples_z, f_IGM_p, alpha0)      # fix f = 0.84, α = 0.2


#   get samples of delta   speed: 40s for 10 samples



def log_likelihood_delta(delta, z, F):      # the value of delta influence F most, bigger value(p) reflects smaller F
    sigma = F/np.sqrt(z)
    c = splc(sigma)
    A = spla(sigma)
    x = (delta**(-3)-c)**2/18/sigma**2
    return  np.log(A*delta**(-3)*np.exp(-x) + ie)       # the accurate value of c is not such influencial to results


def prior_delta(delta):
    if 1/400 < delta < 20:     # expectation of delta must be smaller than 1 for the prior
        return 0.0
    return -np.inf

def log_probability_delta(delta, z, F):
    lp = prior_delta(delta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_delta(delta, z, F)


#method 1

pos_delta = [0.9] + 1e-4 * np.random.randn(nsamples, 1)
nwalkers_d = pos_delta.shape[0]

samples_delta = []

for i in range(nsamples):
    sampler_delta = emcee.EnsembleSampler(nwalkers_d, ndim, log_probability_delta, args=(samples_z[i], F0))     #fix F = 0.2
    sampler_delta.run_mcmc(pos_delta, discard+1)
    samples_delta.append(sampler_delta.get_chain(discard = discard, flat = True)[i])

samples_delta = np.array(samples_delta).reshape(nsamples, 1)

print('delta mean:{0:.3f}, min:{1:.3f}, max:{2:.3f}'.
      format(np.mean(samples_delta),np.min(samples_delta),np.max(samples_delta)))

samples_cosmic = samples_delta * cosmic_average




# get samples of dm_host

def log_likelihood_host(dm_host, sigma_host, emu):
    return  -0.5*(np.log(2*np.pi)+2*np.log(dm_host*sigma_host)+((np.log(dm_host/emu))**2)/sigma_host**2)

def prior_host(dm_host):
    if 1 < dm_host < 2000:
        return 0.0
    return - np.inf

def log_probability_host(dm_host, sigma_host, emu):
    lp = prior_host(dm_host)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_host(dm_host, sigma_host, emu)


pos_host = [100] + 1e-4 * np.random.randn(20, 1)

sampler_host = emcee.EnsembleSampler(nwalkers, ndim, log_probability_host, args=(sigma_host0, emu0))    #fix sigma = 1, emu = 100

sampler_host.run_mcmc(pos_host, nsamples+discard, progress=True)

samples_host = sampler_host.get_chain(discard=discard, thin=thin, flat=True)

print("host mean:{0:.1f}, min:{1:.1f}, max:{2:.1f}".
      format(np.mean(samples_host), np.min(samples_host), np.max(samples_host)))


samples_frb = samples_host/(1+samples_z) + samples_cosmic


#save samples data

my_workbook = xlwt.Workbook()

sheet = my_workbook.add_sheet('test_sheet')

sheet.write(0, 0, "z")
sheet.write(0, 1, "dm_frb")     #sheet.write(row行,col列,value)#

for i in range(samples_z.shape[0]):
    sheet.write(i+1, 0, samples_z[i][0])
for j in range(samples_frb.shape[0]):
    sheet.write(j+1, 1, samples_frb[j][0])
my_workbook.save(r'F:\pythonProject1\data_save\100samples_data3.xlsx')

