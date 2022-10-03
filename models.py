#models.py

import matplotlib.pyplot as plt
import numpy as np

#Constants
pi=np.pi;
gamma=267*1e+6;
mu0=4*pi*1e-7;
h_bar=1.055*1e-34;

#NMR parameters
J_HH = 17.2;
delta_CS = 0.11*1e-6;

#Dipole-dipole parameters
r_hh=1.8e-10;
b_hh1=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3;
b_hh2=-2*pi*10000;


#Correlation times
#t_cf=5*1e-12
#t_cb=40*1e-9


#Spectral density J(w,t_c)
def J1(w,t_c): 
    return t_c/(1+w**2*t_c**2)

#R1 relaxation in free
def R1_f(B,t_c,J):
    w=B*gamma
    return 3/10*b_hh1**2*(J(w,t_c)+4*J(2*w,t_c))

#R1 relaxation in bound
def R1_b(B,t_c,J):
    w=B*gamma
    return 1/10*b_hh1**2*(J(0,t_c)+3*J(w,t_c)+6*J(2*w,t_c))

#Total T1 relaxation model 1
def T1_t_model1(B,t_cf,t_cb,pbA):
    w=B*gamma

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    return 1/(3/10*b_hh1**2*(J(w,t_cf)+4*J(2*w,t_cf))+1/10*pbA*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb)))

     #Total T1 relaxation model from Ziqing Wang 2021
def R1_t_model2(B,t_cf,t_cb,pbA):
    w=B*gamma
    b_hh=-mu0/pi*h_bar*gamma**2/r_hh**3;

    def J(w,t_c): 
        return 2/5*t_c/(1+w**2*t_c**2)

    return 1/4*1/4*3/32*b_hh**2*(J(w,t_cf)+4*J(2*w,t_cf))+pbA*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb))
