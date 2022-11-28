#models.py

import matplotlib.pyplot as plt
import numpy as np

#Constants
pi=np.pi
gamma=267*1e+6
mu0=4*pi*1e-7
h_bar=1.055*1e-34

#NMR parameters
#Ala-Gly at pH=7.2:
#J_HH = 17.2
#delta_CS = 0.11*1e-6
#Citrate at pH=7.2:
J_HH = 15.1
delta_CS = 0.11*1e-6

#Dipole-dipole parameters
r_hh=179*1e-12
b_hh1=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3
b_hh2=-2*pi*10000


#Correlation times
#t_cf=5*1e-12
#t_cb=40*1e-9


#Spectral density J(w,t_c)
def J1(w,t_c): 
    return t_c/(1+w**2*t_c**2)

####################################################
# T1 relaxation models 
####################################################

#R1 relaxation mechanisms
def R1_dd_intra(B,t_c):
    w=B*gamma
    return 3/10*b_hh1**2*(J1(w,t_c)+4*J1(2*w,t_c))

def R1_dd_inter(A,B,t_c):
    w=B*gamma
    return A*(J1(0,t_c)+3*J1(w,t_c)+6*J1(2*w,t_c))

def R1_csa(d_csa,B,t_c):
    w=B*gamma
    return 1/15*(d_csa*w)**2*J1(w,t_c)

####################################################
# T1 relaxation models using real protein concentrations
####################################################
def R1_t_model1_CH2(B,c,t_cf,t_cb,p,A): #Total T1 relaxation Model 1 for CH2 group - intra in free, intra and inter in bound
    return R1_dd_intra(B,t_cf)+c*p*(R1_dd_intra(B,t_cb)+R1_dd_inter(A,B,t_cb))

def R1_t_model1v2_CH2(B,c,t_cf,t_cb,d_csa,p,A): #Total T1 relaxation Model 1 for CH2 group - intra in free, intra and inter in bound
    return R1_dd_intra(B,t_cf)+R1_csa(d_csa,B,t_cf)+c*p*(R1_dd_intra(B,t_cb)+R1_dd_inter(A,B,t_cb)+R1_csa(d_csa,B,t_cb))

def R1_t_model2_CH2(B,c,t_cf,t_cb,t_cm,s2,p,A): #Total T1 relaxation Model 2 for CH2 group - intra in free, intra and inter in bound with order parameter S^2 and correlation time t_cm
    t_cb1=(t_cb*t_cm)/(t_cb+t_cm)
    return R1_dd_intra(B,t_cf)+c*p*(s2*(R1_dd_intra(B,t_cb)+R1_dd_inter(A,B,t_cb))+(1-s2)*(R1_dd_intra(B,t_cb1)+R1_dd_inter(A,B,t_cb1)))

def R1_t_model2v2_CH2(B,c,t_cf,t_cb,t_cm,d_csa,s2,p,A): #Total T1 relaxation Model 4 for CH2 group - intra in free and intra in bound with order parameter S^2 and correlation time t_cm
    t_cb1=(t_cb*t_cm)/(t_cb+t_cm)
    return R1_dd_intra(B,t_cf)+R1_csa(d_csa,B,t_cf)+c*p*(s2*(R1_dd_intra(B,t_cb)+R1_dd_inter(A,B,t_cb)+R1_csa(d_csa,B,t_cb))+(1-s2)*(R1_dd_intra(B,t_cb1)+R1_dd_inter(A,B,t_cb1)+R1_csa(d_csa,B,t_cb1)))
####################################################
# Tlls relaxation models 
####################################################

#Rs relaxation mechanisms

def Rs(Rslow,B,t_c):
    w=B*gamma
    theta=np.arctan((2*pi*J_HH)/(w*delta_CS))/2
    return (20+2*np.cos(4*theta)- np.sqrt(2*(81-80*np.cos(4*theta)+np.cos(8*theta))))/60*R1_dd_intra(B,t_c)+Rslow

def Rs_csa(d_csa,B,t_c):
    w=B*gamma
    return 2/9*(d_csa*w)**2*(1-np.cos(109)**2)*(2*J1(0,t_c)+3*J1(w,t_c))

####################################################
# Tlls relaxation models using real protein concentrations
####################################################
def Rs_t_model1_CH2(B,c,t_cf,t_cb,p,Rslow,A): 

    return Rs(Rslow,B,t_cf)+p*c*(Rs(Rslow,B,t_cb)+R1_dd_inter(A,B,t_cb))

def Rs_t_model1v2_CH2(B,c,t_cf,t_cb,p,Rslow,d_csa,A): 

    return Rs(Rslow,B,t_cf)+Rs_csa(d_csa,B,t_cf)+p*c*(Rs(Rslow,B,t_cb)+R1_dd_inter(A,B,t_cb)+Rs_csa(d_csa,B,t_cb))

def Rs_t_model2_CH2(B,c,t_cf,t_cb,t_cm,p,Rslow,s2,A): 
   
    t_cb1=(t_cb*t_cm)/(t_cb+t_cm)

    return Rs(Rslow,B,t_cf)+p*c*(s2*(Rs(Rslow,B,t_cb)+R1_dd_inter(A,B,t_cb))+(1-s2)*(Rs(Rslow,B,t_cb1)+R1_dd_inter(A,B,t_cb1)))

def Rs_t_model2v2_CH2(B,c,t_cf,t_cb,t_cm,p,Rslow,d_csa,s2,A): 
   
    t_cb1=(t_cb*t_cm)/(t_cb+t_cm)

    return Rs(Rslow,B,t_cf)+Rs_csa(d_csa,B,t_cf)+p*c*(s2*(Rs(Rslow,B,t_cb)+R1_dd_inter(A,B,t_cb)+Rs_csa(d_csa,B,t_cb))+(1-s2)*(Rs(Rslow,B,t_cb1)+R1_dd_inter(A,B,t_cb1)+Rs_csa(d_csa,B,t_cb1)))


