#fitting_multi.py

import matplotlib.pyplot as plt
import numpy as np
import pandas
from models import *
import lmfit
from lmfit import Model, Parameters, minimize, report_fit
import corner
from tqdm import tqdm

data_init = pandas.read_csv("./ALA-GLY and HSA FD.csv")
data_list = data_init.values.tolist()
data_ar=np.array(data_list)
#print(data_ar[:,[0,1]])

x=data_ar[:,0]
#y=1/data_ar[:,[5,9,13,17]]
#y=data_ar[:,[5,9,13,17]] #T1 relaxation data
y=data_ar[:,[7,11,15,19]] #T1 relaxation data
#y_err=data_ar[:,[6,10,14,18]]/(data_ar[:,[5,9,13,17]]**2)
#y_err=data_ar[:,[6,10,14,18]] #T1 relaxation data error
y_err=data_ar[:,[8,12,16,20]] #T1 relaxation data error
#print(y_err)

print('data\n')
data_y=np.transpose(y)
data_err=np.transpose(y_err)
#y=np.vsplit(y,1)
print(data_y)
print(np.shape(data_y))

def model_dataset(params, i, x):
    #Calculate R1 lineshape from parameters for data set.
    t_cf = params[f't_cf_{i+1}']
    t_cb = params[f't_cb_{i+1}']
    t_cm = params[f't_cm_{i+1}']
    s2 = params[f's2_{i+1}']
    pb = params[f'pb_{i+1}']
    #A = params[f'A_{i+1}']
    #pbA = params[f'pbA_{i+1}']
    Rslow = params[f'Rslow_{i+1}']
    #Rslow2 = params[f'Rslow2_{i+1}']
    return 1/(Rs_t_model5_CH2(x,t_cf,t_cb,t_cm,pb,Rslow,s2))

def residual(params, x, data, eps = None):
    #Calculate total residual for fits of R1 to several data sets.
    ndata, _ = data.shape
    resid = 0.0*data[:]

    # make residual per data set
    if eps is None:
        for i in range(ndata):     
            resid[i, :] = data[i, :] - model_dataset(params, i, x)
         # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()
     # make residual per data set
    for i in range(ndata):     
        resid[i, :] = (data[i, :] - model_dataset(params, i, x))/eps[i, :]
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

params = Parameters()
for iy, y in enumerate(data_y):
    params.add(f't_cf_{iy+1}', value=0.05*1e-9, min=1e-14, max=1e-7)
    params.add(f't_cb_{iy+1}', value=40*1e-9, min=1e-14, max=1e-7)
    params.add(f't_cm_{iy+1}', value=1*1e-9, min=1e-14, max=1e-7)
    params.add(f's2_{iy+1}', value=0.1, min=1e-16, max=1)
    params.add(f'pb_{iy+1}', value=1e-4, min=1e-9, max=1)    
    #params.add(f'A_{iy+1}', value=1e+6, min=1e+3, max=1e+16)
    #params.add(f'pbA_{iy+1}', value=1e+6, min=1e+3, max=1e+16)
    params.add(f'Rslow_{iy+1}', value=1e-1, min=1e-4, max=1e+3)
    #params.add(f'Rslow2_{iy+1}', value=1e+2, min=1e-4, max=1e+3)

for iy in (2, 3, 4):
    params[f't_cf_{iy}'].expr = 't_cf_1'
    params[f't_cb_{iy}'].expr = 't_cb_1'
    params[f't_cm_{iy}'].expr = 't_cm_1'
    params[f's2_{iy}'].expr = 's2_1'
    #params[f'A_{iy}'].expr = 'A_1'
    params[f'Rslow_{iy}'].expr = 'Rslow_1'
    #params[f'Rslow2_{iy}'].expr = 'Rslow2_1'

#out = minimize(residual, params, args=(x, data_y),method='differential_evolution')
out = minimize(residual, params, args=(x, data_y),method='leastsq')
report_fit(out.params)

print('-------------------------------')
print('Parameter    Value       Stderr')
for name, param in out.params.items():
    print(f'{name:7s} {param.value:.2E} {param.stderr:.2E}')

###############################################################################
# Plot the data sets and fits
fig1 = plt.figure(figsize=(10, 6),dpi=100)
gs = fig1.add_gridspec(1,4, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
x1 = np.logspace(-3, 1.5, 100)
for i in range(4):
    y_fit = model_dataset(out.params, i, x1)
    axs[i].semilogx(x1, y_fit, '-')
    axs[i].errorbar(x, data_y[i, :], yerr = data_err[i, :],marker='o',capsize=5)

axs[0].set_ylabel('Tlls relaxation time, s')

for ax in axs.flat:
    ax.set(xlabel='B, T')

fig1.suptitle('Tlls relaxation FD')
#ax.set_xlabel('B, T')
#ax.set_ylabel('T1 relaxation time, s')
fig1.savefig('Fitting.png')
plt.show()

