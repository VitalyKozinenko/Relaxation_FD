#fitting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas
from models import *
import lmfit
from lmfit import Model, Parameters, minimize, fit_report
import corner
from tqdm import tqdm

data = pandas.read_csv("./DSS and HSA FD.csv")
data_list = data.values.tolist()
data_ar=np.array(data_list)
#print(data_ar[:,[0,1]])

x=data_ar[:,0]
#y=data_ar[:,1]
y=1/data_ar[:,3]
#y_err=data_ar[:,2]
y_err=data_ar[:,4]/(data_ar[:,3]**2)

gmodel = Model(R1_t_model2)
params = Parameters()
params.add('t_cf', value=0.05*1e-9, min=1e-14, max=1e-7,vary=True)
#params.add('t_cTMS', value=1*1e-12, min=1e-14, max=1e-7,vary=True)
params.add('t_cb', value=40*1e-9, min=1e-14, max=1e-7,vary=True)
params.add('pbA', value=10*1e+6, min=100, max=1e+12,vary=True)
#params.add('pb', value=0.005, min=1e-6, max=1,vary=True)
#params.add('A', value=4*1e+6, min=100, max=1e+12,vary=True)

def residual(pars, x, data = None, eps = None):
    parvals = pars.valuesdict()
    t_cf = parvals['t_cf']
    #t_cTMS = parvals['t_cTMS']
    t_cb = parvals['t_cb']
    #pb = parvals['pb']
    #A = parvals['A']
    pbA = parvals['pbA']

    #model=(R1_t_model1(x,t_cf,t_cb,pbA))
    #model=R1_t_model2(x,t_cf,t_cTMS,t_cb,pbA)
    #model=R1_t_model3(x,t_cf,t_cb,pb,A)
    model=(R1_t_model1_CH2(x,t_cf,t_cb,pbA))
    #model=(R1_t_model2_CH2(x,t_cf,t_cb,pb,A))
    #model=(R1_t_model3_CH2(x,t_cf,t_cb,pb))
    if data is None:
        return model
    if eps is None:
        return model - data
    return (model-data) / eps

mi = minimize(residual, params, args=(x, y, y_err))
#print(mi.params.pretty_print())
#mi1 = minimize(residual, mi.params, args=(x, y, y_err),method='leastsq')
print(fit_report(mi))

res = minimize(residual,args=(x, y, y_err), method='emcee', burn=500, steps=2000, thin=50, params=mi.params, progress=True)
fig1=corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
fig1.set_tight_layout(True)
fig1.savefig('Correlation.png')

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
fig2.savefig('Fitting.png')
plt.show()