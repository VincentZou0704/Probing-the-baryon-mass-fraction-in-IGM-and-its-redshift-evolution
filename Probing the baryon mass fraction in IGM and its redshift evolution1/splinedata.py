import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


ie = 1e-11
z_build = np.linspace(ie, 9, 5000)
omega_m = 0.315
scale = 1000


def devide(n, nscale):
    ret = np.linspace(ie, n, nscale)  # return a shape of (nscale, len(n))
    return ret


def splinec0():
    def excel_one_line_to_list_c(i):
        df = pd.read_excel(r"F:\pythonProject1\data_z\sigma_c0.xlsx", usecols=[i], names=None)
        df_li = np.array(df.values.tolist()).flatten()
        return df_li
    a = excel_one_line_to_list_c(0)
    b = excel_one_line_to_list_c(1)
    return IUS(a,b)


def splineA():
    def excel_one_line_to_list_a(i):
        df = pd.read_excel(r"F:\pythonProject1\data_z\spline_A.xlsx", usecols=[i], names=None)
        df_li = np.array(df.values.tolist()).flatten()
        return df_li
    a = excel_one_line_to_list_a(0)
    b = excel_one_line_to_list_a(1)
    return IUS(a,b)


def splinehez(omega_m):
    sca = z_build / (scale - 1)
    z = devide(z_build, scale)
    He_z = (1 + z) / np.sqrt(omega_m * (1 + z) ** 3 + 1 - omega_m)
    fun_sum = (sca * He_z).sum(axis=0)
    return IUS(z_build, fun_sum)


def splinedcz(omega_m):
    sca = z_build / (scale - 1)
    z = devide(z_build, scale)
    Dc_z = 1 / np.sqrt(omega_m * (1 + z) ** 3 + 1 - omega_m)
    fun_sum = (sca * Dc_z).sum(axis=0)
    return IUS(z_build, fun_sum)


def splineh_gamma(omega_m):
    sca = z_build / (scale - 1)
    z = devide(z_build, scale)
    Hgamma_z = 1 / np.sqrt(omega_m * (1 + z) ** 3 + 1 - omega_m)/(1+z)**2
    fun_sum = (sca * Hgamma_z).sum(axis=0)
    return IUS(z_build, fun_sum)