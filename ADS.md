from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayes_opt import util
import logfile
import ADS_Bayes
import os
import rawdata

bds = {  'C1':(1,5),
         'C2':(0.1,3),
         'C3':(0.1,3),
         'K':(0.1,0.5),
         'Lp':(0.1,3),
         'Ls':(0.1,3)}

os.environ['HPEESOF_DIR'] = r'D:\ADS2020'
installation_path = os.environ['HPEESOF_DIR']
path_list = [installation_path + '\\bin', installation_path + '\\circuit\\lib.win32_64',
             installation_path + '\\adsptolemy\\lib.win32_64']
os.environ['PATH'] += os.pathsep + os.pathsep.join(path_list)
os.environ["SIMARCH"] = 'win32_64'

optimizer = BayesianOptimization(
    f=ADS_Bayes.blackbox,
    pbounds=bds,
    verbose=2,
    random_state=123,)
# optimizer.set_gp_params(normalize_y=True, alpha=2.5e-3, n_restarts_optimizer=20)

X_init = np.array([[4.217023951974466, 1.6485045258510396, 2.829956342952011, 0.5, 0.788330712887237, 0.7294616817113199]])
para = []
para.append(X_init.tolist()[0])
# sim_r=[]
logfile.changenetlist(para[0])

# utility = UtilityFunction(kind="poi", kappa=0.01, xi=0.1)
# m52 = ConstantKernel(1.0) * Matern(length_scale=1, nu=2.5)
# gpr = GaussianProcessRegressor(kernel=m52, alpha=0.1 ** 2).fit(X_init, y)
# a = ADS_Bayes.blackbox()
# sim_r.append(a)
# optimizer._gp.fit(X_init,np.array([a]))
optimizer.set_gp_params(normalize_y=True, alpha=2.5e-3, n_restarts_optimizer=20)

# print(optimizer.res)
# optimizer.set_bounds(new_bounds={"x": (-2, 3)})
optimizer.maximize(init_points=1,
                 n_iter=200,
                 acq='ei',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.1)
# for _ in range(150):
#     next_point_to_probe = optimizer.suggest(utility)
#     # util.acq_max(utility,)
#     para.append(list(next_point_to_probe.values()))
#     # print(next_point_to_probe)
#     # print(list(next_point_to_probe.values()))
#     # print(para[-1],para[-2])
#     logfile.changenetlist(para[-1])
#     a = ADS_Bayes.blackbox()
#     # print(np.array([a]))
#     sim_r.append(a)
#     # print(optimizer.res)
#     try:
#         optimizer.register(params=next_point_to_probe, target=a)
#     except KeyError:
#         continue


    # print('Next Point:{}'.format(next_point_to_probe))
    # print(optimizer.max)

print(optimizer.max)
# print(para[-1])
# para.append(list(optimizer.max['params'].values()))
# print(next_point_to_probe)
# print(list(next_point_to_probe.values()))
# # print(para[-1],para[-2])
# logfile.changenetlist(list(optimizer.max['params'].values()))
# ADS_Bayes.blackbox()
# x,y,z = rawdata.getdata()
# o = -20 * np.log10((x[0]**2 + x[1]**2)**0.5)
# p = -20 * np.log10((y[0]**2 + y[1]**2)**0.5)
# q = -20 * np.log10((z[0]**2 + z[1]**2)**0.5)
# print('S21@2.7GHz = {}'.format(o))
# print('S11@3.3GHz = {}'.format(p))
# print('S11@3.8GHz = {}'.format(q))
# print(list(optimizer.max['params'].values()))


