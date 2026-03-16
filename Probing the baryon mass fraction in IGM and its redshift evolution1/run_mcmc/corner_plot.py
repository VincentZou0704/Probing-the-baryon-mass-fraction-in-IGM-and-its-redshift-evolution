import corner
import pandas as pd
from frb_mcmc.settings import *
import matplotlib.pyplot as plt
from pylab import mpl

def getsamples(i):
    df = pd.read_excel(r'F:\pythonProject1\process\mcmc\mcmc_a0_4p_100.xlsx', usecols=[i], header=None)
    df_li = np.array(df.values.tolist()).flatten()
    return df_li

fiducial = [F0, 0., sigma_host0, emu0]
labels = [r'$F$', r'$\alpha$', '$σ_{host}$', 'exp(μ)']
# r'$F$', r'$f_{IGM,0}$', r'$\alpha$', '$σ_{host}$', 'exp(μ)'

flat_samples = []
for i in range(len(labels)):
    flat_samples.append(getsamples(i))
    print(np.min(np.array(getsamples(i))), np.max(np.array(getsamples(i))))

flat_samples = np.array(flat_samples).transpose()
print(flat_samples.shape)

# range_real= [1, 1, 1, (0.6,2), 1]
# range_x = [(0.12,0.3), (0.1,0.38), 1, 1]

if __name__ == '__main__':
    plt.figure()
    # mpl.rcParams['font.size'] = 18
    fig = corner.corner(
        flat_samples, quantiles=[0.1587, 0.5, 0.8413],
        # range=range_x,
        truths=fiducial,
        label_kwargs={'fontsize': 22},
        labels=labels, smooth=1, smooth1d=1,
        show_titles = True
    )

    # font = {'size': 40}
    # plt.title(r"$N = 100$" + '\n' + r'$\alpha = 0.2$', font, x = 0., y = 3.5)

    # plt.savefig(r"F:\pythonProject1\process\figures\a2_4p_100.pdf")

    plt.show()
