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
r_hh=179*1e-12;
b_hh1=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3;
b_hh2=-2*pi*10000;


#Correlation times
#t_cf=5*1e-12
#t_cb=40*1e-9


#Spectral density J(w,t_c)
def J1(w,t_c): 
    return t_c/(1+w**2*t_c**2)

####################################################
# T1 relaxation models 
####################################################

#R1 relaxation in free
def R1_f(B,t_c,J):
    w=B*gamma
    return 3/10*b_hh1**2*(J(w,t_c)+4*J(2*w,t_c))

#R1 relaxation in bound
def R1_b(B,t_c,J):
    w=B*gamma
    return 1/10*b_hh1**2*(J(0,t_c)+3*J(w,t_c)+6*J(2*w,t_c))

#########################
# Models for CH3
#########################
     
def R1_t_model1(B,t_cf,t_cb,pbA): #Total T1 relaxation Model 1 from Ziqing Wang 2021 - intra dipole in free and inter dipole in bound
    w=B*gamma
    b_hh=-mu0/pi*h_bar*gamma**2/r_hh**3
    S2tms=1/4
    Amethyl=1/4*3/32*b_hh**2

    def J(w,t_c): 
        return 2/5*t_c/(1+w**2*t_c**2)

    return S2tms*Amethyl*(J(w,t_cf)+4*J(2*w,t_cf))+pbA*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb))

def R1_t_model2(B,t_cf,t_cTMS,t_cb,pbA): #Total T1 relaxation Model 2 from Ziqing Wang 2021 - TMS rotation in free form
    w=B*gamma
    b_hh=-mu0/pi*h_bar*gamma**2/r_hh**3
    S2tms=1/4
    Amethyl=1/4*3/32*b_hh**2

    t_c1=(t_cf*t_cTMS)/(t_cf+t_cTMS)

    def J(w,t_c): 
        return 2/5*t_c/(1+w**2*t_c**2)

    return Amethyl*(S2tms*(J(w,t_cf)+4*J(2*w,t_cf))+(1-S2tms)*(J(w,t_c1)+4*J(2*w,t_c1)))+pbA*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb))

def R1_t_model3(B,t_cf,t_cb,pb,A): #Total T1 relaxation Model 3 from Ziqing Wang 2021 - finite concentration of bound
    w=B*gamma
    b_hh=-mu0/pi*h_bar*gamma**2/r_hh**3
    S2tms=1/4
    Amethyl=1/4*3/32*b_hh**2

    def J(w,t_c): 
        return 2/5*t_c/(1+w**2*t_c**2)

    return (1-pb)*S2tms*Amethyl*(J(w,t_cf)+4*J(2*w,t_cf))+pb*(A*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb))+S2tms*Amethyl*(J(w,t_cb)+4*J(2*w,t_cb)))

#########################
# Models for CH2
#########################

def R1_t_model1_CH2(B,t_cf,t_cb,pb): #Total T1 relaxation Model 3 for CH2 group - intra in free and bound
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    return 3/10*b_hh**2*(J(w,t_cf)+4*J(2*w,t_cf))+pb*3/10*b_hh**2*(J(w,t_cb)+4*J(2*w,t_cb))

def R1_t_model2_CH2(B,t_cf,t_cb,pbA): #Total T1 relaxation Model 1 for CH2 group - intra in free and inter in bound
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    return 3/10*b_hh**2*(J(w,t_cf)+4*J(2*w,t_cf))+pbA*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb))


def R1_t_model3_CH3(B,t_cf,t_cb,pb,A): #Total T1 relaxation Model 2 for CH2 group - intra in free and inter and intra in bound
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    return 3/10*b_hh**2*(J(w,t_cf)+4*J(2*w,t_cf))+pb*(A*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb))+3/10*b_hh**2*(J(w,t_cb)+4*J(2*w,t_cb)))


def R1_t_model4_CH2(B,t_cf,t_cb,t_cm,s2,pb): #Total T1 relaxation Model 4 for CH2 group - intra in free and intra in bound with order parameter S^2 and correlation time t_cm
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3

    t_cb1=(t_cb*t_cm)/(t_cb+t_cm)

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    return 3/10*b_hh**2*(J(w,t_cf)+4*J(2*w,t_cf))+pb*3/10*b_hh**2*(s2*(J(w,t_cb)+4*J(2*w,t_cb))+(1-s2)*(J(w,t_cb1)+4*J(2*w,t_cb1)))

def R1_model_freeCH2(B,t_cf): #Total T1 relaxation Model 3 for CH2 group
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    return 3/10*b_hh**2*(J(w,t_cf)+4*J(2*w,t_cf))


####################################################
# T1 relaxation models using real protein concentrations
####################################################
def R1_t_model5_CH2(B,c,t_cf,t_cb,t_cm,s2,p): #Total T1 relaxation Model 4 for CH2 group - intra in free and intra in bound with order parameter S^2 and correlation time t_cm
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3

    t_cb1=(t_cb*t_cm)/(t_cb+t_cm)

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    return 3/10*b_hh**2*(J(w,t_cf)+4*J(2*w,t_cf))+c*p*3/10*b_hh**2*(s2*(J(w,t_cb)+4*J(2*w,t_cb))+(1-s2)*(J(w,t_cb1)+4*J(2*w,t_cb1)))

####################################################
# Tlls relaxation models 
####################################################

def Rs_t_model1_CH2(B,t_cf,t_cb,pbA,Rslow): 
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3
    theta=np.arctan((2*pi*J_HH)/(w*delta_CS))/2

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    def R1(t_c):
        return 3/2*t_c*b_hh**2

    def Rs(Rslow,t_c):
        return (20+2*np.cos(4*theta)- np.sqrt(2*(81-80*np.cos(4*theta)+np.cos(8*theta))))/60*R1(t_c)+Rslow

    return Rs(Rslow,t_cf)+pbA*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb))

def Rs_t_model2_CH2(B,t_cf,t_cb,pb,Rslow):
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3
    theta=np.arctan((2*pi*J_HH)/(w*delta_CS))/2

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    def R1(t_c):
        return 3/2*t_c*b_hh**2

    def Rs(Rslow,t_c):
        return (20+2*np.cos(4*theta)- np.sqrt(2*(81-80*np.cos(4*theta)+np.cos(8*theta))))/60*R1(t_c)+Rslow

    return Rs(Rslow,t_cf)+pb*3/10*b_hh**2*(J(w,t_cb)+4*J(2*w,t_cb))

def Rs_t_model3_CH2(B,t_cf,t_cb,A,pb,Rslow1): 
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3
    theta=np.arctan((2*pi*J_HH)/(w*delta_CS))/2

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    def R1(t_c):
        return 3/10*b_hh**2*(J(w,t_c)+4*J(2*w,t_c))

    def Rs(Rslow,t_c):
        return (20+2*np.cos(4*theta)- np.sqrt(2*(81-80*np.cos(4*theta)+np.cos(8*theta))))/60*R1(t_c)+Rslow

    return Rs(Rslow1,t_cf)+pb*(A*(J(0,t_cb)+3*J(w,t_cb)+6*J(2*w,t_cb))+ Rs(Rslow1,t_cb))

def Rs_t_model4_CH2(B,t_cf,t_cb,t_cm,pb,Rslow,s2): 
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3
    theta=np.arctan((2*pi*J_HH)/(w*delta_CS))/2
    t_cb1=(t_cb*t_cm)/(t_cb+t_cm)

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    def R1(t_c):
        return 3/2*t_c*b_hh**2

    def Rs(Rslow,t_c):
        return (20+2*np.cos(4*theta)- np.sqrt(2*(81-80*np.cos(4*theta)+np.cos(8*theta))))/60*R1(t_c)+Rslow

    return Rs(Rslow,t_cf)+pb*3/10*b_hh**2*(s2*(J(w,t_cb)+4*J(2*w,t_cb))+(1-s2)*(J(w,t_cb1)+4*J(2*w,t_cb1)))

def Rs_t_model5_CH2(B,t_cf,t_cb,t_cm,pb,Rslow,s2): 
    w=B*gamma
    b_hh=-mu0/(4*pi)*h_bar*gamma**2/r_hh**3
    theta=np.arctan((2*pi*J_HH)/(w*delta_CS))/2
    t_cb1=(t_cb*t_cm)/(t_cb+t_cm)

    def J(w,t_c): 
        return t_c/(1+w**2*t_c**2)

    def R1(t_c):
        return 3/10*b_hh**2*(J(w,t_c)+4*J(2*w,t_c))

    def Rs(Rslow,t_c):
        return (20+2*np.cos(4*theta)- np.sqrt(2*(81-80*np.cos(4*theta)+np.cos(8*theta))))/60*R1(t_c)+Rslow

    return Rs(Rslow,t_cf)+pb*(s2*(Rs(Rslow,t_cb)+3/10*b_hh**2*(J(w,t_cb)+4*J(2*w,t_cb)))+(1-s2)*(Rs(Rslow,t_cb1)+3/10*b_hh**2*(J(w,t_cb1)+4*J(2*w,t_cb1))))
