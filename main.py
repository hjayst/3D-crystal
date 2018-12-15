import numpy as np
import scipy
from scipy.integrate import simps
import scipy.io
from config import *
import time
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool
from Bessel_micromotion import *
from functools import partial

indmax = 16 #number of threads

data = scipy.io.loadmat("fid_N50_2.mat")
kDel = 2*2*np.pi/wavelength
omega_K = data["zeta_k"].flatten()
omega_K = omega_K*RF/2.0
#print("Input mode is {0}".format(omega_K))


C = data["C"]
B_new = data["B_2n"]
#N = np.shape(B_new)[0]
#print("number of ions {0}".format(N))
#tau = 300*10**(-6)
kDel_ =  kDel
kDel = np.array([0,0,kDel])#only consider the z
#RF = 40*10**6*2*np.pi
##print np.shape(omega_K)
#omega_K = omega_K*RF/2.0
"""
b_z = np.sum(b.T.reshape(np.shape(omega_K)[0],N,3)*kDel.reshape(1,1,-1),axis=2)#[num_mode, num_ions]
Etak = np.sqrt(const_hbar/(2*const_m*omega_K))#[num_mode]
gk = b_z.T*Etak.reshape(1,-1)#[num_ions, num_modes]
"""

b_jk = (np.sum(np.transpose(C,(0,2,1)).reshape(len(C),len(omega_K),N,3)*kDel.reshape(1,1,1,-1),axis=3))/float(kDel_)
b_jk = np.transpose(b_jk,(0,2,1))
#print("b_jk is {0}".format(b_jk))
Etak = np.sqrt(const_hbar/(2*const_m*omega_K))*float(kDel_)#[num_mode]
#modespan = ((max(omega_K)-min(omega_K)))/2000

#modespan = 0
##print("mode span is {0} MHz".format(modespan/10**6))


#muDelAll = np.arange(2.5*10**7,3.5*10**7,2*np.pi*10**3)
#muDelAll = np.arange((min(omega_K)),(max(omega_K)+10**6),2*np.pi*10**3)#Extend the range by 1MHz.

muDelAll = np.arange(0,(max(omega_K)+10**6),2*np.pi*10**3)#Extend the range by 1MHz.

#print("Number of mu is {0}".format(np.shape(muDelAll)))
#Add the additional term of phiM
Dimension_R = (const_Cou/(const_m*RF**2))**(1/3.) #Take back the dimension of r
R0 = B_new[:,np.shape(B_new)[-1]/2].reshape(N,3,-1)*Dimension_R
R1 = 2*B_new[:,np.shape(B_new)[-1]/2+1:].reshape(N,3,-1)*Dimension_R
"""
phi0 = R0z = np.sum(R0*kDel.reshape(1,-1),axis=1)#The first order phase
phi1 = R1z = np.sum(R1*kDel.reshape(1,-1),axis=1)
"""
#print ("micromotion term is {0}".format(R1.shape))
phi1 = R1z = np.sum(R1*kDel.reshape(1,-1,1),axis=1)
#for test:
#phi1 = phi1[:,0][:,np.newaxis]

phi0 = np.array([0]*N)


#Bessel_n = Bessel(200,phi1)


##print Bessel_n
#np.save("Bessel_n", Bessel_n)
##print("Finished saving Bessel function")

nsegAll = [10]
ith = 9
jth = 14

#print("positions is {0}".format(R0))

posdif = np.sqrt(np.sum((R0[ith,:]-R0[jth,:])**2))
#print("Position difference for {0}th and {1}th is {2}".format(ith,jth,posdif))

phiM = phi0
fidAll = []
#print("The micromotion phase is {0} and {1}".format(phi1[ith], phi1[jth]))
Detuning = 2*np.pi*10**2 #Add an additional detuning in case of mode-mu==0.
muDelAll = muDelAll + Detuning
np.save("muDelAll.out",muDelAll)
#muDelAll = [max(omega_K)*0.9]*16

#muDelAll = [0.9*np.max(omega_K)]

error = 10**-5

#print("b_jk is {0}".format(b_jk))
#print("Etak is {0}".format(Etak))

ind = range(indmax)
def main(ind, ith, jth, nsegAll, muDelAll, phiM, phi1,b_jk):
	numMu = np.shape(muDelAll)[0]
	for nseg in nsegAll:
	    ##print nseg
	    time_seg = time.time()
	    indexlist = np.arange(ind,numMu,indmax)
	    ##print indexlist
	    for id in indexlist:
		if id%100 == 0:
		    print("nseg {0}, id {1}".format(nseg,id))
		mu = muDelAll[id]	
		#print("mu is {0}".format(mu))
		time_loop = time.time()
		#A_k = gk[:,:,np.newaxis]*alpha*-1.0j
#(omega_K, b) = findmodeAll(iterR)
#(omega_K, b) = findmodeAll(iterR)
		"""
		alpha1_ki = alpha_Fun(nseg, tau, mu, ith, phiM[ith], omega_K, N, Bessel_n)
		alpha2_ki = alpha_Fun(nseg, tau, mu, jth, phiM[jth], omega_K, N, Bessel_n)
		
		"""
		##print("phi1 is {0} and {1}".format(phi1[ith],phi1[jth]))
		##print("tau si {0}".format(tau))
		##print("ith, jth is {0},{1}".format(ith,jth))
		##print("mu is {0}".format(mu))
		##print("RF is {0}".format(RF))
		#b_jk = b_jk[2,:,:][np.newaxis,:,:]
		alpha1_ki = alpha_function(ith, nseg, tau, mu, phi1[ith,:], omega_K, RF, b_jk, error)
		alpha2_ki = alpha_function(jth, nseg, tau, mu, phi1[jth,:], omega_K, RF, b_jk, error)
		##print ("alpha2_ki is {0}".format(alpha2_ki))
		A_k1 = Etak[:,np.newaxis]*alpha1_ki*-1.0j
		A_k2 = Etak[:,np.newaxis]*alpha2_ki*-1.0j
		
		##print("A_k2 is {0}".format(A_k2))
		betak = 1/(np.tanh((omega_K*const_hbar)/(2*const_k*Temperature))).reshape(-1)
		
		
		
		M = 0
		for k in range(np.shape(omega_K)[0]):
		    alpha_ki = A_k1[k,:].reshape(1,-1)
		    alpha_ki_dag = np.conjugate(alpha_ki.T)
		    alpha_kj =  A_k2[k,:].reshape(1,-1)
		    alpha_kj_dag = np.conjugate(alpha_kj.T)
		    M += betak[k]*(alpha_ki_dag.dot(alpha_ki)+alpha_kj_dag.dot(alpha_kj))

		M = np.real((M.T+M)/2)
		time0 = time.time()
		print("M is {0}".format(M))
		#phi = np.imag(phifun(nseg, tau, mu, phiM[ith], phiM[jth], ith, jth, omega_K,Bessel_n))
		phi = phiFun_micromotion(ith, jth, nseg, tau, mu, phi1[ith,:], phi1[jth,:], omega_K, RF, b_jk, error)
		Gamma = np.sum((Etak*Etak).reshape(-1,1,1)*phi,axis=0)
		Gamma = (Gamma.T+Gamma)/2.0
		print("Gamma is {0}".format(Gamma))
		time0 = time.time()
		(OmE, OmUnit) = scipy.linalg.eig(M,Gamma)
		indx_Omega = np.argmin(np.abs(OmE))
		
		
		OmUnit = OmUnit[:,indx_Omega].reshape(-1,1)
		Unit = np.dot(OmUnit.reshape(1,-1),np.dot(Gamma,OmUnit.reshape(-1,1)))
		#time0 = time.time()
		if Unit > 0:
		    Omega_i = np.sqrt(np.pi/(4*Unit))*OmUnit
		    Flag = 1
		else:
		    Omega_i = np.sqrt(-np.pi/(4*Unit))*OmUnit
		    Flag = -1

		fidelity = Avg_FID(Omega_i, tau, muDel=mu, alpha_ki = A_k1, alpha_kj=A_k2, Phi=Gamma, ith=ith,jth=jth, betak=betak, Flag=Flag)
		##print fidelity
		#file.write(id)
		#file.write("\n")
		file = open("result/"+str(nseg)+"_fidelity_"+str(ind)+".out","a")
		file.write(str(float(fidelity))+"\n"+str(float(mu))+"\n")
		file.close()
		#file.close()
		print("fidelity is {0}".format(fidelity))
		##print("fidelity here is {0}".format(fidelity))
		#time0 = time.time()
		#fidnsegAll.append(float(fidelity))
		print("time in one loop is {0}".format(time.time()-time_loop))
	    print("time in one segment is {0}".format(time.time()-time_seg))
	    #fidAll.append(fidnsegAll)

pool = Pool(indmax)
ids = range(indmax)
#print nsegAll
#print ith
#print jth
#print muDelAll
#print phi0
#print("phi1 is {0}".format(phi1)) 
#print("b_jk is {0}".format(b_jk))
func1 = partial(main, ith=ith, jth=jth, nsegAll=nsegAll, muDelAll=muDelAll, phiM=phi0, phi1=phi1,b_jk=b_jk)

#main(ind[0], ith, jth, nsegAll, muDelAll, phi0, phi1,b_jk)
#print type(ids)

pool.map(func1, ids)
#func1(0)

	
