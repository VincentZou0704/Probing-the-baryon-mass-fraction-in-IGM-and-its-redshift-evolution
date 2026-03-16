import emcee
import corner
import get_samples_inverseF
from frb_mcmc.initialization import *
from scipy import optimize
import xlwt


def log_likelihood(theta, dm_frb, z):
    alpha, sigma_host,  emu = theta
    sca = (dm_frb*(1+z)) / (scale-1)
    res = sca * (likelihood_all(devide(dm_frb*(1+z)-ie, scale),
                                sigma_host, emu, dm_frb, z, f_IGM_p, alpha, F0).sum(axis=0))     #integrate (sum to speed up)
    res = np.abs(res) + ie  # to avoid bad things
    return np.sum(np.log(res))


# emcee

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


labels = [r'$\alpha$','$σ_{host}$', 'exp(μ)']
fiducial = [alpha0, sigma_host0, emu0]
ndim = len(labels)
nwalkers1, nwalkers2 = 2*ndim, 5*ndim
nll = lambda *args: -log_likelihood(*args)
initial = np.array(fiducial) + 1e-4*np.random.randn(3)

while True:
    z = get_samples_inverseF.get_samples_z(3)
    dm_frb = get_samples_inverseF.get_samples_frb(z)
    soln = optimize.minimize(nll, initial, args=(dm_frb, z))
    alphax, sigmax, emux = soln.x[0], soln.x[1], soln.x[2]
    print('alphax：', alphax, '\t', 'sigmax:', sigmax, '\t', 'emux:', emux)

    if (alpha0-0.025) <= alphax <= (alpha0+0.025):

        pos = fiducial + 1e-4 * np.random.randn(nwalkers1, ndim)

        sampler = emcee.EnsembleSampler(
            nwalkers1, ndim, log_probability, args=(dm_frb, z)
        )
        sampler.run_mcmc(pos, 500, progress=True)
        flat_samples = sampler.get_chain(discard=100, flat=True)

        alpha = np.percentile(flat_samples[:,0], 50)
        sigma= np.percentile(flat_samples[:, 1], 50)
        emu = np.percentile(flat_samples[:, 2], 50)
        print('α:', alpha,'\t', 'σ:', sigma,'\t', 'emu:', emu)

        if (alpha0-0.005) <= alpha <= (alpha0+0.02) and sigma > 0.75 and emu > 75:

            pos = fiducial + 1e-4 * np.random.randn(nwalkers2, ndim)

            sampler = emcee.EnsembleSampler(
                nwalkers2, ndim, log_probability, args=(dm_frb, z)
            )

            sampler.run_mcmc(pos, 3000, progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=10, flat=True)

            alpha_lower, alpha_upper = np.percentile(flat_samples[:, 0], 15.87) - alpha0, \
                                       np.percentile(flat_samples[:, 0], 84.13) - alpha0
            sigma_lower, sigma_upper = np.percentile(flat_samples[:, 1], 15.87) - sigma_host0, \
                                       np.percentile(flat_samples[:, 1], 84.13) - sigma_host0
            emu_lower, emu_upper = np.percentile(flat_samples[:, 2], 15.87) - emu0, \
                                   np.percentile(flat_samples[:, 2], 84.13) - emu0
            for j in range(ndim):
                print(np.percentile(flat_samples[:, j], 50))
            if alpha_lower*alpha_upper <0 and sigma_lower*sigma_upper <0 and emu_upper*emu_lower <0:
                break



if __name__ == '__main__':
    # save samples

    def save_samples(z0, frb0):
        my_workbook = xlwt.Workbook()
        sheet = my_workbook.add_sheet('sample1')
        sheet.write(0, 0, "z0")
        sheet.write(0, 1, "frb0")  # sheet.write(row行,col列,value)

        for i in range(z0.shape[0]):
            sheet.write(i + 1, 0, z0[i])
        for j in range(z0.shape[0]):
            sheet.write(j + 1, 1, frb0[j])
        my_workbook.save(r'F:\pythonProject1\data_save\a0_3p_300samples.xlsx')

    # save mcmc results

    def save_mcmc_result(flat_samples):
        my_workbook = xlwt.Workbook()
        sheet = my_workbook.add_sheet('mcmc_result')
        for i in range(flat_samples.shape[0]):
            for j in range(flat_samples.shape[1]):
                sheet.write(i, j, flat_samples[i][j])
        my_workbook.save(r'F:\pythonProject1\data_save\mcmc_data\mcmc_a0_3p_300.xlsx')

    save_mcmc_result(flat_samples)

    #plot corner


    plt.figure()
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles=[0.1587, 0.5, 0.8413], show_titles=True, title_kwargs={"fontsize": 12}, smooth = 1, smooth1d = 3
    )
    plt.savefig(r"F:\pythonProject1\figure\a0_3p_300samples")
    plt.show()


    # print result
    print_result = open('../100samples_1.txt', mode='a', encoding='utf-8')

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [15.87, 50, 84.13])
        q = np.diff(mcmc)
        print(labels[i] + "={0:.2f}".format(mcmc[1]), '\t', '68%: ', "up:{1:.2f}, down:{0:.2f}".format(q[0], q[1]),
              file=print_result)
    print('\n', file=print_result)
    print_result.close()