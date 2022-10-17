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
#y=data_ar[:,[7,11,15,19]] #Ts relaxation data
y=data_ar[:,[5,9,13,17,7,11,15,19]] #T1 and Ts relaxation data
#y_err=data_ar[:,[6,10,14,18]]/(data_ar[:,[5,9,13,17]]**2)
#y_err=data_ar[:,[6,10,14,18]] #T1 relaxation data error
#y_err=data_ar[:,[8,12,16,20]] #Ts relaxation data error\
y_err=data_ar[:,[6,10,14,18,8,12,16,20]] #T1 and Ts relaxation data error
#print(y_err)

print('data\n')
data_y=np.transpose(y)
data_err=np.transpose(y_err)
#y=np.vsplit(y,1)
#print(data_y)
#print(np.shape(data_y))

conc = np.array([50, 100, 200, 300])/20000


def model_dataset_Ts(params, x, c):
    #Calculate R1 lineshape from parameters for data set.
    t_cf = params['t_cf']
    t_cb = params['t_cb']
    #t_cm = params['t_cm']
    #s2 = params['s2']
    #pb = params[f'pb_{i+1}']
    p = params['p']
    #d = params[f'd_{i+1}']
    A = params['A']
    #pbA = params[f'pbA_{i+1}']
    Rslow = params['Rslow']
    #Rslow2 = params['Rslow2']
    return 1/(Rs_t_model9_CH2(x,c,t_cf,t_cb,p,Rslow,A))

def model_dataset_T1(params, x, c):
    #Calculate R1 lineshape from parameters for data set.
    t_cf = params['t_cf']
    t_cb = params['t_cb']
    #t_cm = params['t_cm']
    #s2 = params['s2']
    #pb = params[f'pb_{i+1}']
    p = params['p']
    #d = params[f'd_{i+1}']
    A = params['A']
    #pbA = params[f'pbA_{i+1}']
    return 1/(R1_t_model8_CH2(x,c,t_cf,t_cb,p,A))


def residual(params, x,c, data, eps = None):
    #Calculate total residual for fits of R1 to several data sets.
    ndata, _ = data.shape
    resid = 0.0*data[:]

    # make residual per data set
    if eps is None:
        for i in range(0,4):     
            resid[i, :] = data[i, :] - model_dataset_T1(params, x, c[i])
        for i in range(4,8):     
           resid[i, :] = data[i, :] - model_dataset_Ts(params, x, c[i-4])
         # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()
     # make residual per data set
    for i in range(0,4):     
        resid[i, :] = (data[i, :] - model_dataset_T1(params, x, c[i]))/eps[i, :]
    for i in range(4,8):   
        resid[i, :] = (data[i, :] - model_dataset_Ts(params, x, c[i-4]))/eps[i, :]
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

params = Parameters()
params.add('t_cf', value=20*1e-12, min=1e-12, max=1e-9)
params.add('t_cb', value=40*1e-9, min=1*1e-12, max=1e-7)
#params.add('t_cm', value=2*1e-9, min=1e-11, max=1e-8)
#params.add('s2', value=0.01, min=1e-3, max=1)
#params.add(f'pb_{iy+1}', value=1e-3, min=1e-9, max=1)
params.add('p', value=1e-1, min=1e-3, max=1)
#params.add(f'd_{iy+1}', value=1e-2, min=1e-9, max=1)        
params.add('A', value=1e+6, min=1e+5, max=1e+9)
#params.add(f'pbA_{iy+1}', value=1e+6, min=1e+3, max=1e+16)
params.add('Rslow', value=1e-1, min=1e-3, max=1)
#params.add('Rslow2', value=1e+3, min=1e-2, max=1e+6)
#params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))



#out1 = minimize(residual, params, args=(x,conc, data_y,data_err),method='differential_evolution')
out1 = minimize(residual, params, args=(x,conc, data_y,data_err),method='leastsq')
report_fit(out1)

#print('-------------------------------')
#print('Parameter    Value       Stderr')
#for name, param in out1.params.items():
#    print(f'{name:7s} {param.value:.2E} {param.stderr:.2E}')
"""
out1 = minimize(residual,args=(x,conc, data_y,data_err), method='emcee', burn=100, steps=2000,nwalkers=2000,thin=200,params=out.params,is_weighted=True, progress=True)
print(len(out1.var_names))
fig1 = plt.figure(figsize=(30, 6),dpi=100)
gs = fig1.add_gridspec(1,len(out1.var_names), wspace=0)
axs= gs.subplots(sharey=True)
plt.tight_layout()
for i in range(len(out1.var_names)):
    axs[i].hist(out1.flatchain.loc[:,out1.var_names[i]],20, ec="yellow", fc="green")
    axs[i].set_xlabel(out1.var_names[i])
#fig1=corner.corner(out1.flatchain, labels=out1.var_names, truths=list(out1.params.valuesdict().values()))
#fig1.set_tight_layout(True)
#fig1.savefig('Correlation.png')

#plt.plot(out1.acceptance_fraction, 'o')
#plt.xlabel('walker')
#plt.ylabel('acceptance fraction')
#plt.show()

print('median of posterior probability distribution')
print('--------------------------------------------')
lmfit.report_fit(out1.params)
"""
###############################################################################
# Plot the data sets and fits
fig2 = plt.figure(figsize=(10, 6),dpi=100)
gs = fig2.add_gridspec(2,4, wspace=0.3,hspace=0)
axs= gs.subplots(sharex=True, sharey=False)
x1 = np.logspace(-3, 1.5, 100)
for i in range(0,4):
    y_fit1 = model_dataset_T1(out1.params, x1,conc[i])
    axs[0,i].semilogx(x1, y_fit1, '-')
    axs[0,i].errorbar(x, data_y[i, :], yerr = data_err[i, :],marker='o',capsize=5)
    axs[0,i].set_ylim(0.7, 1.9)
    y_fit2 = model_dataset_Ts(out1.params, x1,conc[i])
    axs[1,i].semilogx(x1, y_fit2, '-')
    axs[1,i].errorbar(x, data_y[i+4, :], yerr = data_err[i+4, :],marker='o',capsize=5)
    axs[1,i].set_ylim(3, 15)


axs[0,0].set_ylabel('T1 relaxation time, s')
axs[1,0].set_ylabel('Tlls relaxation time, s')

for ax in axs.flat:
    ax.set(xlabel='B, T')


fig2.suptitle('Tlls relaxation FD')
#ax.set_xlabel('B, T')
#ax.set_ylabel('T1 relaxation time, s')
fig2.savefig('Fitting.png')


plt.show()