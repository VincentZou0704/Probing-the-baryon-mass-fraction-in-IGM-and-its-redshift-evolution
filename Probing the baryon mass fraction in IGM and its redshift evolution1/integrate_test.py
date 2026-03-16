import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


dmfrb = np.array([361.42, 589.27, 363.6, 338.7, 593.1])  #此项中仅包含DM_host和DM_cosmic, 338.7, 593.1
dmism = np.array([41.89419211467203, 42.39581689047655, 56.815577390267244, 38.063401552135424, 55.93063340127531])  #从数据库中得出 , 38.063401552135424, 55.93063340127531
dmhalo = np.array([50]*5)
dm0 = dmfrb - dmism - dmhalo
z0 = np.array([0.3214, 0.4755, 0.291, 0.1178, 0.522])  # ,0.1178,0.522
h0_P = 66.74
ob_P = 0.0486
f_IGM_p = 0.84
c0 = np.array([1.9239, 1.7477, 1.9700, 2.3714, 1.7067])  #固定F=0.2解出的与红移一一对应的C0值, 2.3713713713713718, 1.7067067067067068
a = np.array([0.3802, -1.9527, 2.5970, 1.5797])   #多项式拟合的C0关于sigma_2的参数

coefficient = 21*3*10**5/64/np.pi/6.67/1.67/3.0857**2    # 1MPC = 3.0857*10**22 m 还要乘H0,omega,f_IGM才是完整的系数

def dm_c_a(ob_h, z):   #dm_cosmic_average,h0 hided in omega_b
    ob = ob_h*70/h0_P
    return coefficient*f_IGM_p*(z+z**2/2)*ob*h0_P  #如何把Omega和h70一起放进去？

def devide(n, nscale):
    ret = np.linspace(0.1, n, nscale)     #return a shape of (nscale, len(n))
    return ret

def like_host(dm_h, s_h, emu): #已检验,函数正确
    return  np.exp(-0.5*(np.log(2*np.pi)+2*np.log(dm_h*s_h)+((np.log(dm_h/emu))**2)/s_h**2))


def like_cos(dm_h, dm, z, ob_h, F, c): #已检验,函数正确
    dm_c = dm - dm_h/(1+z)
    delta = dm_c/dm_c_a(ob_h, z)
    sigma_2 = (F/np.sqrt(z))**2
    x = (delta**(-3)-c)**2/18/sigma_2
    return  delta**(-3)*np.exp(- x)       #c值对结果影响不大


def combi(dm_h, s_h, emu, dm, z, ob_h, F, c): #已检验,函数正确
    return like_host(dm_h, s_h, emu)*like_cos(dm_h, dm, z, ob_h, F, c)

scale = 1000

def log_likelihood(theta, dm, z, c):
    F, ob_h, s_h, emu = theta
    sca = dm / scale
    n = dm*(1+z)-0.1
    print(n)
    res = sca * (combi(devide(n, scale), s_h, emu, dm, z, ob_h, F, c).sum(axis=0))  #test in integrate_test
    print(res)
    res = np.array(res) + 1 * 10 ** (-5)
    print(np.sum(np.log(res)))

log_likelihood([0.4,0.056,0.88,68.2],dm0,z0,c0)