from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt



dmfrb = np.array([557.00, 536.00, 348.80, 362.16, 103.50, 589.00,
                  364.55, 760.80, 340.05, 332.63, 592.60, 504.13,
                  507.90, 297.50, 380.25, 577.80, 413.52])  # delete 181030[4] and 190611[9]
dmism = np.array([157.60, 136.53, 168.73, 41.45, 40.16, 41.98,
                  56.22, 36.74, 37.81, 56.60, 55.37, 38.00,
                  44.22, 33.75, 27.35, 36.19, 126.49])  # 从数据库中得出

dmhalo = np.array([50]*17)
dm0 = dmfrb - dmism - dmhalo    # 此项中仅包含DM_host和DM_cosmic
dm0 = list(dm0) + [114 - 37] + [1205 - 60]  # FRB 20171020A and rFRB 20190520B
dm0 = np.array(dm0)

print(dm0)