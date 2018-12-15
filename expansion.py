import numpy as np
import copy
import functools
from scipy.special import *
import scipy.io 
from itertools import product
import time
import numexpr as ne


def multicos(mu, Cof, omega_k, ta, tb):
    # Consider the integration with the shape Id = Exp[i mu t] Cos[A1t] Cos[A2t] Cos[A3t] ... Cos[ANt]Exp[I*wk*t]
    # And suppose we have got the coefficient matrix with the shape [N]
    #1st to (n-2)th is the coefficient A1,A2,A3,...,AN.
    #(n)th is the coefficient mu
    #omega_k is a list of mode 
    #Cof = np.array([mu,Cof]).flatten()
    Cof = np.array(Cof).flatten()
    omega_k = np.array([omega_k]).reshape(1,-1)
    Cof1 = Cof
    A = list(product([1,-1],repeat=len(Cof)))
    A = np.array(A)
    n = len(A)
    ##print A
    Coe = np.sum(A*Cof1.reshape(1,-1),axis=1).reshape(-1,1)
    res1 = 1.0/n *  ComputeExp(Coe, mu,omega_k, ta, tb)
    ##print res1
    #res2 = -1.0/n * 1.0j * ComputeExp(-Coe,mu,omega_k, ta, tb)
    ###print res2
    res = np.sum(res1,axis=0)
    return res

def ComputeExp(Cof, mu, omega_k, ta, tb):
    #Here compute the integral exp(i ta A1 A2 ... An)
    return 1.0j*(np.exp(1.0j*ta*(Cof+mu+omega_k))-np.exp(1.0j*tb*(Cof+mu+omega_k)))/(Cof+mu+omega_k)


def NextLchild(father, childindx, res):
    M = copy.deepcopy(father[-1])
    #childindx.append(0)
    M[childindx[-1]] += 1
    childindx.append(childindx[-1])
    res.append(M)
    father.append(M)
    return (father, childindx, res)

def NextRbrother(father, childindx, res):
    
    M = copy.deepcopy(father[-1])
    M[childindx[-1]] += 1
    res.append(M)
    childindx[-1] += 1
    return (father,childindx, res)


def finduncle(father, childindx):
    childindx.pop()
    father.pop()
    if father == []:
        return False
    else:
        childindx[-1] += 1
        return (father,childindx)


def single_dfs_Integral(phi1, RF, Cmag, mu, omega_k, ta, tb, error, flag, I0=False):
    #print flag
    MaxBreadth = len(phi1)
    res = []
    childindx = []
    count = 0 #take care of memory
    res.append([0]*MaxBreadth)  
    father = copy.deepcopy(res)
    childindx.append(0)
    maglist = []

    maglist = [functools.reduce(lambda x,y:x*y,jv(res[-1],phi1))]
    RF_list = np.arange(1,len(phi1)+1)*RF
    depth = len(father)
   
    mag = lambda res:Cmag*functools.reduce(lambda x,y:x*y,[2**int(bool(ind))*(1.0j)**ind*jv(ind,phi1[in_]) for in_,ind in enumerate(res)])
    mag_ = lambda res:Cmag*functools.reduce(lambda x,y:x*y,[2**int(bool(ind))*(-1.0j)**ind*jv(ind,phi1[in_]) for in_,ind in enumerate(res)])
    maglist = [mag(res[-1])]
    IntRes = 0
    """
    if type(I0) != bool:
        
        #I1 = IntRes = I0
	I1 = I0
	IntRes = 0
    else:
        I0 = IntRes =1/2.0j*(multicos(mu, res[-1]*RF_list, omega_k, ta, tb)*mag(res[-1])-\
        multicos(-mu, res[-1]*RF_list, omega_k, ta, tb)*mag_(res[-1]))
        return I0
    ##print I1
    ###print childindx
    ###print res
    if flag != 0:
    	I1 = 1/2.0j*(multicos(mu, res[-1]*RF_list, omega_k, ta, tb)*mag(res[-1])-\
    	multicos(-mu, res[-1]*RF_list, omega_k, ta, tb)*mag_(res[-1]))
	IntRes += I1
    """
    I0 = 1/2.0j*(multicos(mu, res[-1]*RF_list, omega_k, ta, tb)*mag(res[-1])-\
    multicos(-mu, res[-1]*RF_list, omega_k, ta, tb)*mag_(res[-1]))

    IntRes = I1 = I0

    while count<100000 and depth >= 1:
        count += 1
        depth = len(father)  
        ##print depth
        if np.max(np.abs(I1/I0)) > error and childindx[-1]<MaxBreadth:
            (father, childindx, res) = NextLchild(father, childindx, res)
            maglist.append(mag(res[-1]))

            ##print res[-1]
            
            I1 = 1/2.0j*(multicos(mu, res[-1]*RF_list, omega_k, ta, tb)*mag(res[-1])-\
            multicos(-mu, res[-1]*RF_list, omega_k, ta, tb)*mag_(res[-1]))
            if np.max(np.abs(I1/I0))>error:
                IntRes += I1

            else:
                (father,childindx) = finduncle(father, childindx)
                res.pop()

        elif np.max(np.abs(I1/I0))<error and childindx[-1]<MaxBreadth: #go to the right child


            (father, childindx, res) = NextRbrother(father, childindx, res)
            maglist.append(mag(res[-1]))

            I1 = 1/2.0j*(multicos(mu, res[-1]*RF_list, omega_k, ta, tb)*mag(res[-1])-\
            multicos(-mu, res[-1]*RF_list, omega_k, ta, tb)*mag_(res[-1]))

            if np.max(np.abs(I1/I0))>error:
                IntRes += I1

            else:
                res1 = finduncle(father, childindx)
                if res1 != False:
                    (father, childindx) = res1
                else:
                    break            
                (father,childindx) = res1
                res.pop()
        else:
            ##print "precise error"
            res1 = finduncle(father, childindx)
            if res1 != False:
                (father, childindx) = res1
            else:
                break 
        ##print("I1 is {0}".format(I1))
    #print("counting number is {0}".format(count))
    return IntRes





def expand_single(mu, ta, tb, omega_k, RF, C, phi1, error):
    res = 0
    n = len(C)//2
    ind = np.argsort(np.abs(np.arange(-n,n+1)))
    for i in ind:
        mag = np.linalg.norm(C[i,:])
        if mag > error:
	    #print("arg of alpha is {0}".format(i))
            if i == n:
		#print("equal arg of alpha is {0}".format(i))
		"""
                I0 = single_dfs_Integral(phi1, RF, C[i,:], mu, omega_k, ta, tb, error,0)
		#print("I0 is {0}".format(I0))
		res = I0
		"""
		I0 = 0
		res += single_dfs_Integral(phi1, RF, C[i,:], mu, omega_k, ta, tb, error,0,I0)           
	    else:	
            	res += single_dfs_Integral(phi1, RF, C[i,:], mu, omega_k+(n-i)*RF, ta, tb, error,1,I0)
    #print("res is {0}".format(res))
    return res

"""
def dualmulticos(mu1, mu2, Cof1, Cof2, omega_k, ta, tb):
    Cof1 = np.array(Cof1).flatten()
    Cof2 = np.array(Cof2).flatten()
    omega_k = np.array([omega_k]).reshape(1,-1)
    A1 = product([1,-1],repeat=len(Cof1))
    A2 = product([1,-1],repeat=len(Cof2))
    
    n = 2**len(Cof1)*2**len(Cof2)
    ##print Coe2
    IntRes=0
    count = 0
    for coe1s in product([1,-1],repeat=len(Cof1)):
        for coe2s in product([1,-1],repeat=len(Cof2)):
            count+=1
            I1 = 1.0/n*ComputeDualExp(np.sum(coe1s*Cof1),np.sum(coe2s*Cof2),mu1,mu2,omega_k,ta,tb)
            IntRes += I1
    
    return IntRes

def ComputeDualExp(Cof1, Cof2, mu1,mu2, omega_k, ta, tb):
    #Calculate the integral x = [ta, tb], y = [ta, x]
    ##print Cof1,Cof2
    #if Cof1+Cof2!=0 and mu1+mu2 != 0:

    if Cof1+Cof2+mu1+mu2!=0:
        return ne.evaluate("((exp(1.0j*(Cof1+Cof2+mu1+mu2)*ta)-\
        exp(1.0j*(Cof1+Cof2+mu1+mu2)*tb))/(Cof1+Cof2+mu1+mu2)+\
        (exp(1.0j*ta*(Cof2+mu2-omega_k))*(-exp(1.0j*ta*(Cof1+mu1+omega_k))+\
        exp(1.0j*tb*(Cof1+mu1+omega_k))))/(Cof1+mu1+omega_k))/(Cof2+mu2-omega_k)")
    else:
        return ne.evaluate("(-exp(1.0j*(Cof1+Cof2+mu1+mu2)*ta)+\
        exp(1.0j*(ta*(Cof2+mu2-omega_k)+tb*(Cof1+mu1+omega_k)))+\
        1.0j*(ta-tb)*(Cof1+mu1+omega_k))/((Cof2+mu2-omega_k)*(Cof1+mu1+omega_k))")

"""

def dualmulticos(mu1, mu2, Cof1, Cof2, omega_k1,omega_k2, ta, tb):
    Cof1 = np.array(Cof1).flatten()
    Cof2 = np.array(Cof2).flatten()
    omega_k1 = np.array([omega_k1]).reshape(1,-1)
    omega_k2 = np.array([omega_k2]).reshape(1,-1)
    A1 = product([1,-1],repeat=len(Cof1))
    A2 = product([1,-1],repeat=len(Cof2))
    
    n = 2**len(Cof1)*2**len(Cof2)
    t1 = time.time()
    A1 = np.sum(np.array(list(product([1,-1],repeat=len(Cof1))))*Cof1,axis=1)
    A2 =np.sum(np.array(list(product([1,-1],repeat=len(Cof2))))*Cof2,axis=1)
    x,y = np.meshgrid(A1,A2)
    Cof1 = (x.flatten()).reshape(-1,1)
    Cof2 = (y.flatten()).reshape(-1,1)
    IntRes = 1.0/n*np.sum(ComputeDualExp(Cof1, Cof2, mu1,mu2, omega_k1,omega_k2, ta, tb),axis=0)
    return IntRes

def ComputeDualExp(Cof1, Cof2, mu1,mu2, omega_k1,omega_k2, ta, tb):
    #Calculate the integral x = [ta, tb], y = [ta, x]
    ##print Cof1,Cof2
    #if Cof1+Cof2!=0 and mu1+mu2 != 0:
    #epsilon = 10**-14
    
    #    #print mu1,mu2
    ##print Cof1,Cof2
    


    C1 = Cof1+Cof2+mu1+mu2+omega_k1-omega_k2
    tx = (tb-ta)/2.0
    ty = (tb+ta)/2.0
    expression1 = -2.0j*np.sinc((C1*tx)/np.pi)*tx*np.exp(1.0j*C1*ty)/(Cof2+mu2-omega_k2)
    expression2 = (np.exp(-1.0j*(Cof1*(ta+tb)+ta*(Cof2-mu2+omega_k2)))*\
    (-np.exp(1.0j*((Cof1)*tb+ta*(mu1+omega_k1)))+\
    np.exp(1.0j*((Cof1)*ta+tb*(mu1+omega_k1)))))/((Cof1-mu1-omega_k1)*(Cof2-mu2+omega_k2))
    expression = expression1 - expression2

    return expression 
    """
    return ne.evaluate("((exp(1.0j*(Cof1+Cof2+mu1+mu2)*ta)-\
    exp(1.0j*(Cof1+Cof2+mu1+mu2)*tb))/(Cof1+Cof2+mu1+mu2+epsilon)+\
    (exp(1.0j*ta*(Cof2+mu2-omega_k))*(-exp(1.0j*ta*(Cof1+mu1+omega_k))+\
    exp(1.0j*tb*(Cof1+mu1+omega_k))))/(Cof1+mu1+omega_k))/(Cof2+mu2-omega_k)")
    """





def double2_dfs_Integral(phi2, Cof1, RF, Cmag, Cmag_, mu, omega_k1,omega_k2, ta, tb, error, I0=False):
    MaxBreadth = len(phi2)
    res = []
    childindx = []
    count = 0 #take care of memory
    res.append([0]*MaxBreadth)  
    father = copy.deepcopy(res)
    childindx.append(0)
    RF_list = np.arange(1,len(phi2)+1)*RF
    depth = len(father)
    Flag = 0
    mag = lambda res:functools.reduce(lambda x,y:x*y,[2**int(bool(ind))*(1.0j)**ind*jv(ind,phi2[in_]) for in_,ind in enumerate(res)])
    mag_ = lambda res:functools.reduce(lambda x,y:x*y,[2**int(bool(ind))*(-1.0j)**ind*jv(ind,phi2[in_]) for in_,ind in enumerate(res)])
    #maglist = [mag(res[-1])]
    def Integrand(res):
        #rint "test Integrand"
        ####print mag(res),mag_(res)
	#print("reslist is {0}".format(res))
        time1 = time.time()
        mag1 = mag(res)
        mag2 = mag_(res)
	#print("Cof1 and res")
	#print(Cof1,res*RF_list)
        ###print Cmag,Cmag_,mag1,mag2
	"""
        Ires = 1.0j/8.0*(dualmulticos(mu, mu, Cof1, res*RF_list, omega_k,ta, tb)*Cmag*mag1-\
        dualmulticos(mu, mu, Cof1, res*RF_list, -omega_k,ta, tb)*Cmag*mag1+\
        dualmulticos(mu, -mu, Cof1, res*RF_list, -omega_k,ta, tb)*Cmag*mag2-\
        dualmulticos(mu, -mu, Cof1, res*RF_list, omega_k,ta, tb)*Cmag*mag2+\
        dualmulticos(-mu, mu, Cof1, res*RF_list, -omega_k,ta, tb)*Cmag_*mag1-\
        dualmulticos(-mu, mu, Cof1, res*RF_list, omega_k,ta, tb)*Cmag_*mag1+\
        dualmulticos(-mu, -mu, Cof1, res*RF_list, omega_k,ta, tb)*Cmag_*mag2-\
        dualmulticos(-mu, -mu, Cof1, res*RF_list, -omega_k,ta, tb)*Cmag_*mag2)
        """
	###print Ires
	#print("Cmag and res")
	#print(Cmag, Cmag_,mag1,mag2)
    	Ires=-1.0/4.0*np.imag(dualmulticos(mu, mu, Cof1, res*RF_list, omega_k1,omega_k2,ta, tb)*Cmag*mag1-\
    	dualmulticos(mu, -mu, Cof1, res*RF_list, omega_k1,omega_k2,ta, tb)*Cmag*mag2-\
    	dualmulticos(-mu, mu, Cof1, res*RF_list, omega_k1,omega_k2,ta, tb)*Cmag_*mag1+\
    	dualmulticos(-mu, -mu, Cof1, res*RF_list, omega_k1,omega_k2,ta, tb)*Cmag_*mag2)
        print time.time()-time1
        return Ires
    
    if type(I0) != bool:
    
        I1 = IntRes = Integrand(res[-1])
        
        if np.max(np.abs(I1/I0)) < error:
            return I1
    else:
        I0 = Integrand(res[-1])

        return I0
    while count<1000000 and depth >= 1:
        ##print np.max(np.abs(I1/I0))
        count += 1
        depth = len(father)  
        #print ("depth 2 is {0}".format(depth))
        if np.max(np.abs(I1/I0)) > error and childindx[-1]<MaxBreadth:
            (father, childindx, res) = NextLchild(father, childindx, res)
            #maglist.append(mag(res[-1]))

            I1 = Integrand(res[-1])
            if np.max(np.abs(I1/I0))>error:
                IntRes += I1

            else:
                (father,childindx) = finduncle(father, childindx)
                res.pop()

        elif np.max(np.abs(I1/I0))<error and childindx[-1]<MaxBreadth: #go to the right child


            (father, childindx, res) = NextRbrother(father, childindx, res)
            #maglist.append(mag(res[-1]))

            I1 = Integrand(res[-1])

            if np.max(np.abs(I1/I0))>error:
                IntRes += I1

            else:
                res1 = finduncle(father, childindx)
                if res1 != False:
                    (father, childindx) = res1
                else:
                    break            
                (father,childindx) = res1
                res.pop()
        else:
            ####print "precise error"
            res1 = finduncle(father, childindx)
            if res1 != False:
                (father, childindx) = res1
            else:
                break 
    #print("counting is {0}".format(count))
    return IntRes

def double1_dfs_Integral(phi1, phi2, RF, Cmag, mu, omega_k1,omega_k2, ta, tb, error,flag, I0 = False):
    MaxBreadth = len(phi1)
    ####print error
    res = []
    childindx = []
    count = 0 #take care of memory
    res.append([0]*MaxBreadth)  
    father = copy.deepcopy(res)
    childindx.append(0)
    #maglist = []
    
    #maglist = [functools.reduce(lambda x,y:x*y,jv(res[-1],phi1))]
    RF_list = np.arange(1,len(phi1)+1)*RF
    depth = len(father)
    Flag = 0
    mag = lambda res:Cmag*functools.reduce(lambda x,y:x*y,[2**int(bool(ind))*(1.0j)**ind*\
    jv(ind,phi1[in_]) for in_,ind in enumerate(res)])
    mag_ = lambda res:Cmag*functools.reduce(lambda x,y:x*y,[2**int(bool(ind))*(-1.0j)**ind*\
    jv(ind,phi1[in_]) for in_,ind in enumerate(res)])
    #maglist = [mag(res[-1])]
    test1 = lambda res:functools.reduce(lambda x,y:x*y,[2**int(bool(ind))*(-1.0j)**ind*\
    jv(ind,phi1[in_]) for in_,ind in enumerate(res)])
    #print("mag and mag_ is {0}".format((test1(res[-1]))))
    
    
    Cof1 = res[-1]*RF_list
    """
    
    if type(I0) != bool:
        I1 = double2_dfs_Integral(phi2, Cof1, RF, mag(res[-1]), mag_(res[-1]), mu, omega_k, ta, tb, error, I0)
    else:
        I0 = double2_dfs_Integral(phi2, Cof1, RF, mag(res[-1]), mag_(res[-1]), mu, omega_k, ta, tb, error)
        #I1 = double2_dfs_Integral(phi2, Cof1, RF, mag(res[-1]), mag_(res[-1]), mu, omega_k, ta, tb, error, I0)
        #I1 = I0
        
        return I0
    """
    if flag==0:
    	if type(I0)==bool:
    		I0 = double2_dfs_Integral(phi2, Cof1, RF, mag(res[-1]), mag_(res[-1]), mu, omega_k1,omega_k2, ta, tb, error)
    		#print("first I0 is {0}".format(I0))
		return I0
    	else:
    		I1 = double2_dfs_Integral(phi2, Cof1, RF, mag(res[-1]), mag_(res[-1]), mu, omega_k1,omega_k2, ta, tb, error,I0)
		IntRes = I1
    else:
	I1 = double2_dfs_Integral(phi2, Cof1, RF, mag(res[-1]), mag_(res[-1]), mu, omega_k1,omega_k2, ta, tb, error,I0)

    	#print("first I1 is {0}".format(I1))
    	IntRes = I1
    while count<1000000 and depth >= 1:
        ##print np.max(np.abs(I1/I0))
        count += 1
        depth = len(father)
        #print ("1 depth {0}".format(depth))
        ####print I1
        if np.max(np.abs(I1/I0)) > error and childindx[-1]<MaxBreadth:
            
            (father, childindx, res) = NextLchild(father, childindx, res)
            #maglist.append(mag(res[-1]))
            
            I1 = double2_dfs_Integral(phi2, res[-1]*RF_list, RF, mag(res[-1]), mag_(res[-1]), mu, omega_k1, omega_k2, ta, tb, error, I0)
            if np.max(np.abs(I1/I0))>error:
                IntRes += I1
            else:
                (father,childindx) = finduncle(father, childindx)
                res.pop()
        elif np.max(np.abs(I1/I0))<error and childindx[-1]<MaxBreadth: #go to the right child
            (father, childindx, res) = NextRbrother(father, childindx, res)
            #maglist.append(mag(res[-1]))
            I1 = double2_dfs_Integral(phi2, res[-1]*RF_list, RF, mag(res[-1]), mag_(res[-1]), mu, omega_k1, omega_k2, ta, tb, error, I0)
            if np.max(np.abs(I1/I0))>error:
                IntRes += I1

            else:
                res1 = finduncle(father, childindx)
                if res1 != False:
                    (father, childindx) = res1
                else:
                    break            
                res.pop()
        else:
            res1 = finduncle(father, childindx)
            if res1 != False:
                (father, childindx) = res1
            else:
                break 
            
    #print("counting is {0}".format(count))
    return IntRes

def expand_double(mu, ta, tb, phi1, phi2, omega_k, RF,C1, C2, error):
    res = 0 
    n1 = len(C1)//2
    n2 = len(C2)//2
    ind1 = np.argsort(np.abs(np.arange(-n1,n1+1)))
    ind2 = np.argsort(np.abs(np.arange(-n2,n2+1)))
    for i in ind1:
        for j in ind2:
            mag = np.linalg.norm(C1[i,:])*np.linalg.norm(C2[j,:])
            if mag > error:
                #print(i,j)
                if i == n1 and j == n2:
                    #print("equal is {0},{1}".format(i,j))
                    #print("greatC is {0}".format(C1[i,:]*C2[j,:]))
                    I0 = double1_dfs_Integral(phi1, phi2, RF, C1[i,:]*C2[j,:], mu, omega_k,omega_k, ta, tb, error,0)
		    I1 =  double1_dfs_Integral(phi1, phi2, RF, C1[i,:]*C2[j,:], mu, omega_k,omega_k, ta, tb, error,0,I0)
                    #print("phi I0 is {0}".format(I0))
		    res += I1

		else:
		    #print("greatC is {0}".format(C1[i,:]*C2[j,:]))
		    ##print("phi is {0}".format(double1_dfs_Integral(phi1, phi2, RF, C1[i,:]*C2[j,:], mu, omega_k+(i-n1)*RF,omega_k+(j-n2)*RF, ta, tb, error, I0)))
                    res += double1_dfs_Integral(phi1, phi2, RF, C1[i,:]*C2[j,:], mu, omega_k+(i-n1)*RF, omega_k+(j-n2)*RF, ta, tb, error,1, I0)
    return res
