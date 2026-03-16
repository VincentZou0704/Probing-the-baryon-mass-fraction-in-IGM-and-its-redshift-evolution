from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import xlwt
from frb_mcmc.settings import *
import pandas as pd
from scipy import optimize


nsamples = 300   # select n samples randomly
min_z = 0.01
min_host = 7.609069432733423
max_host = 1314.221149410639

def devide(_min_, _max_, nscale):
    ret = np.linspace(_min_, _max_, nscale)     # return a shape of (nscale, 1)
    return ret

# get samples of z

def pdf_z(z):
    a, b, c, B, C, eta = 3.4, -0.3, -3.5, 5000, 9, -10
    SFR_z = 0.02*((1+z)**(a*eta)+((1+z)/B)**(b*eta)+((1+z)/C)**(c*eta))**(1/eta)
    H_z = np.sqrt(a1*(1+z)**3+a2)
    return spldc(z)**2*SFR_z/(1+z)/H_z


def cdf_z(x, max_z):    # get cdf of z at x
    piece_z = max_z / (scale - 1)
    k_z = 1 / (piece_z * pdf_z(devide(min_z, max_z, scale)).sum(axis = 0))  # normalization for pdf_z
    piece = x/(scale-1)
    cdf = piece * k_z * (pdf_z(devide(min_z, x, scale)).sum(axis = 0))
    return cdf


def get_samples_z(max_z):
    # max_z = 3       # min_z = 0(set as 1e-6)

    x = np.linspace(min_z, max_z, 1000)
    spl1 = IUS(cdf_z(x,max_z),x)
    z0_u = np.random.rand(nsamples)
    z0 = spl1(z0_u)     # type : numpy.ndarray
    if __name__ == '__main__':
        print('mean of z：', np.mean(z0),'\t','max：',np.max(z0),'\t', 'min：', np.min(z0))
    return z0


# get samples of delta

def pdf_delta(delta, z):      # the value of delta influence F most, bigger value(p) reflects smaller F
    F = F0
    sigma = F/np.sqrt(z)
    c = splc(sigma)
    A = spla(sigma)
    x = (delta**(-3)-c)**2/18/sigma**2
    return  A*delta**(-3)*np.exp(-x)       # the accurate value of c is not such influencial to results


# method 1 (fix max likelihood of delta)
# def get_samples_delta(z0):
#     nll = lambda *args:-pdf_delta(*args)
#     delta0 = []
#     for i in range(nsamples):
#         soln = optimize.minimize(nll, 0.9, args = z0[i])
#         delta0.append(soln.x)
#     delta0 = np.array(delta0).reshape(nsamples,)
#     if __name__ == '__main__':
#         print('delta:', np.mean(delta0),np.max(delta0),np.min(delta0))
#     return delta0


# method 2

def get_delta_1sigma(i):
    li = pd.read_excel(r'F:\pythonProject1\data_z\delta_limit.xlsx', usecols=[i], header=0)
    value = np.array(li.values.tolist()).flatten()
    return value


spl_min_delta = IUS(get_delta_1sigma(0),get_delta_1sigma(3))
spl_max_delta = IUS(get_delta_1sigma(0),get_delta_1sigma(4))


def cdf_delta(x, z, min_delta, max_delta):
    piece_x = (max_delta-min_delta)/(scale-1)
    k_delta = 1/(piece_x * pdf_delta(devide(min_delta, max_delta, scale),z).sum(axis = 0))
    # k_delta = 1
    piece = (x-min_delta)/(scale-1)
    cdf = piece * k_delta * (pdf_delta(devide(min_delta, x, scale),z).sum(axis = 0))
    return cdf


def get_samples_delta(z0):      # 0.9s/it
    delta0 = []
    for i in range(nsamples):
        min_delta = spl_min_delta(z0[i])
        max_delta = spl_max_delta(z0[i])
        x = np.array(list(np.linspace(min_delta, 1, 1000)) +
                     list(np.linspace(1+1e-6, max_delta, 1000)))
        spl2 = IUS(cdf_delta(x, z0[i], min_delta, max_delta), x)
        delta0_u = np.random.rand(1)
        delta0.append(spl2(delta0_u))
    delta0 = np.array(delta0).reshape(nsamples,)
    if __name__ == '__main__':
        print('delta:', np.mean(delta0),np.max(delta0),np.min(delta0))
    return delta0


# end of method 2
# get samples of dm_host

def pdf_host(dm_host, sigma_host, emu):
    return  np.exp(-0.5*(np.log(2*np.pi)+2*np.log(dm_host*sigma_host)+((np.log(dm_host/emu))**2)/sigma_host**2))


def cdf_host(x):
    piece_host = (max_host - min_host) / (scale - 1)
    k_host = 1 / (piece_host * (pdf_host(devide(min_host, max_host, scale), sigma_host0, emu0).sum(axis=0)))  # normalization of pdf_host
    piece = x/(scale-1)
    cdf = piece*((k_host*pdf_host(devide(min_host, x, scale), sigma_host0, emu0)).sum(axis = 0))
    return cdf


def get_samples_host():
    x1 = list(np.linspace(min_host, 200, 2000))
    x2 = list(np.linspace(200.01, 500, 1000))
    x3 = list(np.linspace(500.1, max_host, 500))
    x = np.array(x1 + x2 + x3)
    spl3 = IUS(cdf_host(x),x)
    host0_u = np.random.rand(nsamples)
    host0 = spl3(host0_u)
    if __name__ == '__main__':
         print('emu:', np.percentile(host0,50),'\t', 'sigma_host:',
               np.sqrt(2*np.abs(np.log(np.mean(host0)/np.percentile(host0,50)))), '\t',
               'max:{0:.2f}   min:{1:.2f}'.format(np.max(host0), np.min(host0)))
    return host0


def dm_cosmic_average(z, f, alpha):    # f parameterized as f = f_IGM,0 * (1 + alpha * z/(1+z) )
    # f = 0.84
    # alpha = 0.2
    return coefficient * ob_P * h0_P * f * splz(z) * (1+alpha*z/(1+z))


def get_samples_frb(z0):
    delta0 = get_samples_delta(z0)
    host0 = get_samples_host()
    cosmic0 = dm_cosmic_average(z0, f_IGM_p, alpha0) * delta0
    frb0 = host0/(1+z0) + cosmic0
    if __name__ == '__main__':
        print('dm_frb:',np.mean(frb0),np.max(frb0),np.min(frb0))
    return frb0



if __name__ == '__main__':
    z0 = get_samples_z(3)
    frb0 = get_samples_frb(z0)


    def save_samples(z0, frb0):
        my_workbook = xlwt.Workbook()
        sheet = my_workbook.add_sheet('sample1')
        sheet.write(0, 0, "z0")
        sheet.write(0, 1, "frb0")  # sheet.write(row行,col列,value)

        for i in range(z0.shape[0]):
            sheet.write(i + 1, 0, z0[i])
        for j in range(z0.shape[0]):
            sheet.write(j + 1, 1, frb0[j])
        my_workbook.save(r'F:\pythonProject1\process\samples\a0_4p+f_300samples.xlsx')


    save_samples(z0,frb0)