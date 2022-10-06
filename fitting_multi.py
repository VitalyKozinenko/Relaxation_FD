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
y=data_ar[:,[5,9,13,17]]
#y_err=data_ar[:,[6,10,14,18]]/(data_ar[:,[5,9,13,17]]**2)
y_err=data_ar[:,[6,10,14,18]]
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
    return 1/(R1_t_model4_CH2(x,t_cf,t_cb,t_cm,s2,pb))

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
    params.add(f's2_{iy+1}', value=0.5, min=1e-16, max=1)
    params.add(f'pb_{iy+1}', value=1e-1, min=1e-9, max=1)    
    #params.add(f'A_{iy+1}', value=1e+6, min=1e+3, max=1e+16)

for iy in (2, 3, 4):
    params[f't_cf_{iy}'].expr = 't_cf_1'
    params[f't_cb_{iy}'].expr = 't_cb_1'
    params[f't_cm_{iy}'].expr = 't_cm_1'
    params[f's2_{iy}'].expr = 's2_1'
    #params[f'A_{iy}'].expr = 'A_1'

out = minimize(residual, params, args=(x, data_y),method='leastsq')
report_fit(out.params)

###############################################################################
# Plot the data sets and fits
fig1, ax = plt.subplots()
x1 = np.logspace(-3, 1.5, 100)
for i in range(4):
    y_fit = model_dataset(out.params, i, x1)
    ax.semilogx(x1, y_fit, '-')
    ax.errorbar(x, data_y[i, :], yerr = data_err[i, :],marker='o',capsize=5)

ax.set_title('T1 relaxation FD')
ax.set_xlabel('B, T')
ax.set_ylabel('T1 relaxation time, s')
fig1.savefig('Fitting.png')
plt.show()


"""
#params.add('t_cf', value=0.05*1e-9, min=1e-14, max=1e-7,vary=True)
#params.add('t_cTMS', value=1*1e-12, min=1e-14, max=1e-7,vary=True)
#params.add('t_cb', value=40*1e-9, min=1e-14, max=1e-7,vary=True)
#params.add('pbA', value=10*1e+6, min=100, max=1e+12,vary=True)
#params.add('pb', value=0.005, min=1e-6, max=1,vary=True)
#params.add('A', value=4*1e+6, min=100, max=1e+12,vary=True)

def residual(pars, x, data = None, eps = None):
    parvals = pars.valuesdict()
    t_cf = parvals['t_cf']
    #t_cTMS = parvals['t_cTMS']
    #t_cb = parvals['t_cb']
    #pb = parvals['pb']
    #A = parvals['A']
    #pbA = parvals['pbA']

    #model=(R1_t_model1(x,t_cf,t_cb,pbA))
    #model=R1_t_model2(x,t_cf,t_cTMS,t_cb,pbA)
    #model=R1_t_model3(x,t_cf,t_cb,pb,A)
    #model=(R1_t_model1_CH2(x,t_cf,t_cb,pbA))
    #model=(R1_t_model2_CH2(x,t_cf,t_cb,pb,A))
    #model=(R1_t_model3_CH2(x,t_cf,t_cb,pb))
    #model=(R1_model_freeCH2(x,t_cf))
    if data is None:
        return model
    if eps is None:
        return model - data
    return (model-data) / eps

mi = minimize(residual, params, args=(x, y, y_err))
#print(mi.params.pretty_print())
#mi1 = minimize(residual, mi.params, args=(x, y, y_err),method='leastsq')
print(fit_report(mi))

#res = minimize(residual,args=(x, y, y_err), method='emcee', burn=500, steps=2000, thin=50, params=mi.params, progress=True)
#fig1=corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
#fig1.set_tight_layout(True)
#fig1.savefig('Correlation.png')

fig2, ax = plt.subplots()
#ax.semilogx(x, y, 'o')
ax.errorbar(x, y, yerr = y_err,marker='o',capsize=5)
x1 = np.logspace(-3, 1.5, 100)
ax.semilogx(x1, residual(mi.params,x1), '-', label='best fit')
#ax.semilogx(x1, residual(params,x1), '--', label='initial fit')
ax.set_title('T1 relaxation FD')
ax.set_xlabel('B, T')
ax.set_ylabel('R1 relaxation rate, 1/s')
ax.legend()
ax.set(ylim=(0, 1))
fig2.savefig('Fitting.png')
plt.show()
"""