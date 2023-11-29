import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fitter import Fitter

# Constant 

nu = 0.0003 #cm^2/s
h = 1 #cm

# Fonctiun 

def repartition_speed(u,precision=0.01):
    vitesse = np.sort(u)
    n = len(vitesse)
    v_max = vitesse[n-1]
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

# Pdf of speed

plt.figure(0)
plt.hist(u[1000:30000], bins=100, density=True, label='u')
plt.xlabel('vitesse '+ r'$(cm.s^{-1}$)')
plt.ylabel('pdf')
f = Fitter(u[1000:30000],distributions= ['norm'])
f.fit()
f.summary()
plt.legend()

# cumul of pdf

plt.figure(2)
plt.hist(u[1000:10000], bins=100, density=True, cumulative=True, label='u')

plt.xlabel('vitesse '+ r'$(cm.s^{-1}$)')
plt.ylabel('pdf')
plt.legend()

# Find the distribution of speed
def gaussian_density(x):
    return 1/(np.sqrt(2*np.pi))*np.exp(-0.5*x**2)

liste_x = np.linspace(min(u[1000:10000]), max(u[1000:10000]), 100)

# Plot portion of speed

plt.figure(1)
plt.plot(temps[1000:10000], u[1000:10000], 'r', label='u '+ r'$(cm.s^{-1}$)')
plt.plot(temps[1000:10000], v[1000:10000], 'g', label='v '+ r'$(cm.s^{-1}$)')
plt.plot(temps[1000:10000], w[1000:10000], 'b', label='w '+ r'$(cm.s^{-1}$)')
plt.xlabel('temps (s)')
plt.ylabel('vitesse '+ r'$(cm.s^{-1}$)')
plt.legend()

plt.figure(3)
plt.plot(repartition_speed(u[1000:10000]), pdf_speed(u[1000:10000]), 'r', label='u '+ r'$(cm.s^{-1}$)')

plt.show()




