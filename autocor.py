import numpy as np
import matplotlib.pyplot as plt

def bruit_blanc(temps):
    bruit = np.random.normal(0,1,len(temps))
    return bruit

def signal_sinus(temps,A=1,f=0.2,phi=0):
    signal = A*np.sin(2*np.pi*f*temps+phi)
    return signal

def langevin():
    # This python function write a txt file with a random signal obeying a langevin stochastic equation
    
    dt = 0.01 #the time step of the signal
    Nit = 10000 #the number of time step of the signal 
    
    sigma = 2  #the variance of the langevin signal
    T = 1.  #tha characteristic time of the signal
    mu = 3.  #the mean value of the signal
    
    sigma2 = sigma*sigma  
    k= np.sqrt(2.*dt/T) 

    #the intial condition
    dw = np.random.randn()
    x0 = sigma * np.random.randn()+mu
    t0 = 0.
    
    Y = [t0,x0,dw]
    
    i=0
    x=x0
    t=t0
    while i<Nit:
        i += 1
        dw = k * np.random.randn()
        dx = sigma * dw
        dx += (mu-x)*dt/T
        
        x += dx
        t += dt
        Y = np.vstack( (Y, [t,x,dw]) )  
    #np.savetxt("langevin.txt",Y) 
    return Y
         
#Calcul du coeff d'auto-correlation
def autocorr(u_fluctuations,time):
    Ntot = len(u_fluctuations)  
    var = np.std(u_fluctuations)**2
    N = Ntot//10
    C = np.zeros(N)
    for j in range(N):
        for i in range(Ntot-j):
            C[j] += u_fluctuations[i] * u_fluctuations[i+j]
        C[j] /= (Ntot-j)
    C[j] /= var
    tau=time[0 : N]
    return tau, C

def echelle_integ(tau):
    N = len(tau)
    echelle = np.zeros(N)
    for i in range(N-1):
        echelle[i] = np.sum(tau[i]/(tau[i+1]-tau[i]))
    return echelle
    
# Calcul 

temps = np.arange(0,100,0.01)
signal_sin = signal_sinus(temps)
signal_bruit = bruit_blanc(temps)
signal_langevin = langevin()
