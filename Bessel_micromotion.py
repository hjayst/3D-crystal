import numpy as np
import scipy
from scipy.integrate import simps
import scipy.io
from config import *
import time
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool
import numexpr as ne
from scipy.special import jv
import scipy.io 
from itertools import product
import time
from expansion import *

pool = Pool(16)



def alpha_function(jth, nseg, tau, mu, phi1, omega_k, RF, b_jk, error):
    DeltaT = tau/nseg
    
    if type(omega_k) == int:
        omega_k = np.array([omega_k])
    tseg = np.linspace(0,tau,nseg+1)
    alphaseg_ki = np.zeros([omega_k.size,nseg],dtype=np.complex256)
    for indT in range(nseg):
        
        ta = tseg[indT]
        tb = tseg[indT]+DeltaT
	print(ta,tb)
        alphaseg_ki[:,indT] = expand_single(mu, ta, tb, omega_k, RF, b_jk[:,jth,:], phi1, error)
    return alphaseg_ki



def phiFun_micromotion(ith, jth, nseg, tau, mu, phi1, phi2, omega_k, RF, b_jk, error):
    tseg = np.linspace(0,tau,nseg+1)
    DeltaT = tau/nseg
    phiseg_kij = np.zeros([np.shape(omega_k.flatten())[0],nseg,nseg], dtype=np.complex256)
    for p in range(nseg):
	print("p is {0}".format(p))
        ta = tseg[p]
        tb = tseg[p]+DeltaT
        for q in range(p):
	    print("q is {0}".format(q))

            tc = tseg[q]
            td = tseg[q]+DeltaT
            
            phiseg_kij[:,p,q] = -np.imag(np.conjugate(expand_single(mu, ta, tb, omega_k, RF, b_jk[:,ith,:], phi1, error))*\
            expand_single(mu, tc, td, omega_k, RF, b_jk[:,jth,:], phi2, error)+\
            np.conjugate(expand_single(mu, ta, tb, omega_k, RF, b_jk[:,jth,:], phi2, error))*\
            expand_single(mu, tc, td, omega_k, RF, b_jk[:,ith,:], phi1, error))

	    print(p,q)
	    print("phiseg is {0}".format(phiseg_kij[:,p,q]))
        phiseg_kij[:,p,p]=expand_double(mu, ta, tb, phi1, phi2, omega_k, RF,b_jk[:,ith,:],b_jk[:,jth,:], error)+\
        expand_double(mu, ta, tb, phi2, phi1, omega_k, RF,b_jk[:,jth,:],b_jk[:,ith,:], error)
	print(p,p)
	print("phiseg is {0}".format(phiseg_kij[:,p,p]))
    return np.real(phiseg_kij)






def Avg_FID(Omega_i, tau, muDel, alpha_ki, alpha_kj, Phi, ith, jth, betak, Flag):
    alpha_ki = np.dot(alpha_ki,Omega_i)
    alpha_kj = np.dot(alpha_kj,Omega_i)
    index1 = 2*np.real(np.dot(betak.T,alpha_ki*np.conjugate(alpha_ki)))
    index2 = 2*np.real(np.dot(betak.T,alpha_kj*np.conjugate(alpha_kj)))
    indexCross = 2*np.real(np.dot(betak.T,alpha_kj*np.conjugate(alpha_ki)))
    Gamma1 = np.exp(-index1)
    Gamma2 = np.exp(-index2)
    Gammap=np.exp(-index1-index2-2*(indexCross))
    Gammam=np.exp(-index1-index2+2*(indexCross))
    Phi_ij = np.dot(Omega_i.T,np.dot(Phi, Omega_i))
    DeltaE = 2*np.imag(np.sum(alpha_ki*np.conjugate(alpha_kj)))
    #print(np.shape(Gamma1))
    #fidelity = (4+Flag*2*(Gamma1+Gamma2)*np.sin(2*Phi_ij+DeltaE)+Gammap+Gammam)/10.0
    
    fidelity = (4+2*Flag*(Gamma1*np.sin(2*Phi_ij+DeltaE)+Gamma2*np.sin(2*Phi_ij-DeltaE))+Gammap+Gammam)/10.0
    
    return fidelity


if __name__ == "__main__":
    
    Bessel_n = np.load("Bessel_n.txt.npy")
    #Bessel_n[1] = Bessel(10, )
    #print("phifun(nseg=14, tau=300*10**(-6), mu=10**7, phiM1=1, phiM2=2, ith=1, jth=4, omega_K=10**6, Bessel_n=Bessel_n)")
    #print phifun(nseg=14, tau=300*10**(-6), mu=10**7, phiM1=1, phiM2=2, ith=1, jth=4, omega_K=10**6, Bessel_n=Bessel_n)
    
