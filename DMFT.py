import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import expm

# Simulate two-site DMFT without quantum circuit

#step 1: set val of impurity on site interaction energy
U = 0.0

#step 2: make an intial gues for the val of hybridization parameter
V = 1.0 

#step 3: obtain impurity Green's function from QComp as a fxn of time

h = 1.0 # planck
c = 3.0 * 10**8 # speed of light
impurity = np.array([1,0]) # impurity site spin up
bath = np.array([0,1]) # bath site spin down

def selfConsistency(V, Z, t):
    if(V**2 == Z * t**2):
        return True
    else:
        return False


#SIAM Hamiltonian - half filled case
def ham(U, V): 
    upNum = np.array([[1, 0], [0, 0]]) @ impurity
    downNum = np.array([[0, 0], [1, 0]]) @ impurity

    # c1down = np.dot(np.array([0,0],(1,0)), impurity)
    # a2up = np.dot(np.array([[0, 1], [0, 0]]), np.dot(np.array([[0, 1], [0, 0]]), bath))
    # a2down = 

    return U * upNum * downNum - U/2 * (upNum + downNum) + V * (4.0*h*c)

def impGreenFxn(U, V, tvals, t):
    def timeEv(U, V, t):
        return np.exp(-1j * ham(U,V) * t)
    
    return np.real(np.dot(np.dot(np.array([[0,1],[1,0]]),np.conjugate(timeEv(U,V,t))), np.dot(np.array([[0,1],[1,0]]), timeEv(U,V,t))))

#step 4: using ImpGreenfxn find best fit for params and finding 
def fit_greens_function(t_values, greens_values):
    
    def model(t, a, w, p):
        return a * np.cos(w * t) + (1 - a) * np.cos(p * t)

    params, pop = curve_fit(model, t_values, greens_values)
    return params #[a, w, p]

#step 5: calculate quasiparticle weight
def quasiweight(tvals, t, U, V):
    a, w, p = fit_greens_function(tvals, impGreenFxn(U, V, tvals, t))
    if w == 0:
        w = 1
    if p == 0:
        p = 1

    denom = (V**4 * (a/w**4 + (1-a)/p**4))
    if denom == 0:
        denom = 1

    return denom**(-1)

#step 6: put it all together to reach self consistency

Zvals = []

t = np.linspace(0,10,11, True)

for i in t:
    Z = quasiweight(t, i, U, V)
    U += 2
    Zvals.append(Z)
    
    if selfConsistency(V, Z, i):
        print("Self Consistency has been reached")
        break
    else:
        V = np.sqrt(Z) * i 

Urange = np.linspace(0,10,len(Zvals))

plt.figure()

plt.plot(Urange, Zvals)

plt.title("Quasiparticle weight Z as a fxn of interaction strength U")
plt.xlabel("U")
plt.ylabel("Z")

plt.show()





