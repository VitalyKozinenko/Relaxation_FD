#fitting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas
from models import *
import lmfit
from lmfit import Model, Parameters, minimize
import corner
from tqdm import tqdm

data = pandas.read_csv("./DSS and HSA FD.csv")
data_list = data.values.tolist()
data_ar=np.array(data_list)
print(data_ar[:,[0,1]])

x=data_ar[:,0]
#y=data_ar[:,1]
y=1/data_ar[:,1]
y_err=data_ar[:,2]/(data_ar[:,1]**2)

gmodel = Model(R1_t_model2)
params = Parameters()
params.add('t_cf', value=25*1e-12, min=1*1e-12, max=100*1e-12)
params.add('t_cb', value=40*1e-9, min=1*1e-9, max=100*1e-9)
params.add('pbA', value=15000000, min=100, max=1e+12)

def residual(pars, x, data = None, eps = None):
    parvals = pars.valuesdict()
    t_cf = parvals['t_cf']
    t_cb = parvals['t_cb']
    pbA = parvals['pbA']
    model=R1_t_model2(x,t_cf,t_cb,pbA)
    if data is None:
        return model
    if eps is None:
        return model - data
    return (model-data) / eps

mi = minimize(residual, params, args=(x, y, y_err))
lmfit.report_fit(mi)

#res = minimize(residual,args=(x, y, y_err), method='emcee', burn=300, steps=2000, thin=20, params=mi.params, progress=True)

#result = gmodel.fit(y, params,B=x, weights=yerr,method='emcee')

#corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()),title_kwargs={"fontsize": 18}, smooth=True)

fig, ax = plt.subplots()
#ax.semilogx(x, y, 'o')
ax.errorbar(x, y, yerr = y_err,marker='o',capsize=5)
x1 = np.logspace(-3, 1.5, 100)
ax.semilogx(x1, residual(mi.params,x1), '-', label='best fit')
#ax.semilogx(x, result.init_fit, '--', label='initial fit')
ax.set_title('T1 relaxation FD')
ax.set_xlabel('B, T')
ax.set_ylabel('Total T_1, s')
ax.legend()
plt.show()