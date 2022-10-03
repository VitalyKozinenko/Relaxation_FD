#fitting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas
from models import *
from lmfit import Model, Parameters

data = pandas.read_csv("./DSS and HSA FD.csv")
data_list = data.values.tolist()
data_ar=np.array(data_list)
print(data_ar[:,[0,1]])

x=data_ar[:,0]
#y=data_ar[:,1]
y=1/data_ar[:,1]
yerr=data_ar[:,2]

gmodel = Model(R1_t_model2)
params = Parameters()
params.add('t_cf', value=25*1e-12, min=1*1e-12, max=100*1e-12)
params.add('t_cb', value=35*1e-9, min=1*1e-9, max=100*1e-9)
params.add('pbA', value=15000000, min=100, max=1e+12)
result = gmodel.fit(y, params,B=x, weights=yerr)

print(result.fit_report())

fig, ax = plt.subplots()
#ax.semilogx(x, y, 'o')
ax.errorbar(x, y, yerr = yerr,marker='o',capsize=5)
x1 = np.logspace(-3, 1.5, 100)
ax.semilogx(x1, result.eval(B=x1), '-', label='best fit')
ax.semilogx(x, result.init_fit, '--', label='initial fit')
ax.set_title('T1 relaxation FD')
ax.set_xlabel('B, T')
ax.set_ylabel('Total T_1, s')
ax.legend()
plt.show()

"""
B = np.logspace(-3, 1.5, 100)
t_cf=27*1e-12
t_cb=40*1e-9

fig, ax = plt.subplots()

ax[1].semilogx(B, T1_t(gamma*B,t_cf,t_cb,0.0005*b_hh1**2,J1), label=f"t_c = {t_cf}, s\nt_b = {t_cb}, s\nAb = {0.0005*b_hh1**2:.1e}, Hz^2")
ax[1].scatter(data_ar[:,0],data_ar[:,13])
ax[1].grid()
#ax[1].set_title('T1 relaxation FD')
ax[1].set_xlabel('B, T')
ax[1].set_ylabel('Total T_1, s')
ax[1].legend()
plt.show()                   # Display the plot
"""