from frb_mcmc import splinedata as sd
import numpy as np


h0_P = 67.4
ob_P = 0.0493
f_IGM_p, alpha0, F0, sigma_host0, emu0= 0.84, 0., 0.2, 1, 100
a1 = 0.315
a2 = 1 - a1

c_photon = 2.99792458*10**8
gravity_c = 6.67408*1e-11
m_proton = 1.672621637*1e-27
coefficient = 1e-41*(21*c_photon/64/np.pi/gravity_c/m_proton/3.085677581467**2)


scale = 1000
ie = 1e-11

epsilon = 8.854187817 * 1e-12
h_planck = 6.62607015 * 1e-34
m_electron = 9.10956 * 1e-31
q_electron = 1.602176634 * 1e-19


splz = sd.splinehez(a1)
splc = sd.splinec0()
spla = sd.splineA()
spldc = sd.splinedcz(a1)
splhgamma = sd.splineh_gamma(a1)

if __name__ == '__main__':
    print(coefficient)