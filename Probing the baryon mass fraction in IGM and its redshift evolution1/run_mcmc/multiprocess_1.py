from frb_mcmc.run_mcmc.samples_17 import *


if __name__ == '__main__':
    plt.figure()
    fig = corner.corner(
        flat_samples, labels = labels, show_titles = True,
        quantiles=[0.16, 0.5, 0.84], smooth=1, smooth1d=3
    )
    plt.savefig(r"F:\pythonProject1\process\figures\18samples_3p.png")
    plt.show()


    my_workbook = xlwt.Workbook()
    sheet = my_workbook.add_sheet('mcmc_result')
    for i in range(flat_samples.shape[0]):
        for j in range(flat_samples.shape[1]):
            sheet.write(i, j, flat_samples[i][j])
    my_workbook.save(r'F:\pythonProject1\process\mcmc\18samples_3p.xlsx')


    # print result

    print_result = open(r'F:\pythonProject1\process\19samples.txt', mode='a', encoding='utf-8')

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(labels[i]+"={0:.3f}".format(mcmc[1]), '\t', '68%: ' , "-:{0:.3f}, +:{1:.3f}".format(q[0],q[1]),
              file=print_result)
    print('\n', file=print_result)
    print_result.close()