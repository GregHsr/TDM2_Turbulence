import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fitter import Fitter
import scipy.stats

# Constant 

nu = 0.0003 #cm^2/s
h = 1 #cm

# Fonctiun 

def repartition_speed(u,precision=0.01):
    vitesse = np.sort(u)
    n = len(vitesse)
    v_max = vitesse[-1]
    v_min = vitesse[0]
    repartition = np.linspace(v_min,v_max,int((v_max-v_min)/precision))
    return repartition

def pdf_speed(u,precision=0.01):
    vitesse = np.sort(u)
    n = len(vitesse)
    v_max = vitesse[n-1]
    v_min = vitesse[0]
    repartition = np.linspace(v_min,v_max,int((v_max-v_min)/precision))
    pdf = np.zeros(len(repartition))
    for vit in vitesse:
        for i in range(len(repartition)):
            if vit > repartition[i] and vit < repartition[i+1]:
                pdf[i] += 1
                break
    return pdf

def probabilite(pdf):
    proba = pdf/np.sum(pdf)
    return proba

# Read data from file 'signal_canal3280_48.txt'

data = pd.read_csv('signal_canal3280_48.txt', sep='\t', header=None, names=['iteration', 'temps', 'u', 'v', 'w'])

iteration = np.array(data['iteration'])
temps = np.array(data['temps']) #s
u = np.array(data['u']) #cm/s
v = np.array(data['v'])
w = np.array(data['w'])

print(repartition_speed)

# Calcul

# Mean
mean_u = np.mean(u)
mean_v = np.mean(v)
mean_w = np.mean(w)

print('mean_u = ', mean_u)
print('mean_v = ', mean_v)
print('mean_w = ', mean_w)

# Standard deviation

std_u = np.std(u)
std_v = np.std(v)
std_w = np.std(w)

print('std_u = ', std_u)
print('std_v = ', std_v)
print('std_w = ', std_w)

def loi_normale(x, mean, std):
    return 1/(std*np.sqrt(2*np.pi))*np.exp(-0.5*(x-mean)**2/std**2)

# Plot des vitesses

plt.figure(1)
plt.plot(temps[1000:10000], u[1000:10000], 'r', label='u '+ r'$(cm.s^{-1}$)')
plt.plot(temps[1000:10000], v[1000:10000], 'g', label='v '+ r'$(cm.s^{-1}$)')
plt.plot(temps[1000:10000], w[1000:10000], 'b', label='w '+ r'$(cm.s^{-1}$)')
plt.xlabel('temps (s)')
plt.ylabel('vitesse '+ r'$(cm.s^{-1}$)')
plt.legend()

# Probability
proba_u = probabilite(pdf_speed(u))
print("Somme proba =", sum(proba_u))

# Plot
plt.figure(3)
plt.subplot(311)
plt.plot(repartition_speed(u),proba_u, 'r', label='u')
# Comparaison avec la densité de probabilité gaussienne
x = np.linspace(min(u),max(u), 100)
y = probabilite([loi_normale(abc,mean_u,std_u) for abc in x])
plt.plot(x,y, 'b', label='gaussienne')
plt.legend()
plt.subplot(312)
plt.plot(repartition_speed(v),probabilite(pdf_speed(v)), 'r', label='v')
x = np.linspace(min(v),max(v), 100)
y = probabilite([loi_normale(abc,mean_v,std_v) for abc in x])
plt.plot(x,y, 'b', label='gaussienne')
plt.legend()
plt.subplot(313)
plt.plot(repartition_speed(w),probabilite(pdf_speed(w)), 'r', label='w')
x = np.linspace(min(w),max(w), 100)
y = probabilite([loi_normale(abc,mean_w,std_w) for abc in x])
plt.plot(x,y, 'b', label='gaussienne')
plt.legend()

# Cumulative probability
densite_proba = [sum(proba_u[:i]) for i in range(len(proba_u))]
y_gauss_u = probabilite([loi_normale(abc,mean_u,std_u) for abc in repartition_speed(u)])   
densite_proba_gauss = [sum(y_gauss_u[:i]) for i in range(len(y_gauss_u))]

plt.figure(4)
plt.plot(repartition_speed(u), densite_proba, 'r',label='u')
plt.plot(repartition_speed(u), densite_proba_gauss, 'b',label='gaussienne')
plt.legend()

# Pdf of speed

plt.figure(0)
plt.hist(u[1000:30000], bins=100, density=True, label='u')
plt.xlabel('vitesse '+ r'$(cm.s^{-1}$)')
plt.ylabel('pdf')
f = Fitter(u[1000:30000],distributions= ['norm'])
f.fit()
f.summary()
plt.legend()

# Moyenne probabilité
mean_proba_u = np.sum(repartition_speed(u)*proba_u)
print("mean_proba_y =", mean_proba_u)
mean_proba_v = np.sum(repartition_speed(v)*probabilite(pdf_speed(v)))
print("mean_proba_y =", mean_proba_v)
mean_proba_w = np.sum(repartition_speed(w)*probabilite(pdf_speed(w)))
print("mean_proba_y =", mean_proba_w)

plt.show()


