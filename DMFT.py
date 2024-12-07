import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import expm

# -----------------------------Simulate two-site DMFT without quantum circuit-------------------------------------#

#step 1: set val of impurity on site interaction energy
Urange = np.array([0,2,4,6,8,10])

#step 2: make an intial guess for the val of hybridization parameter
V_initial = 10.0 

#step 3: obtain impurity Green's function from QComp as a fxn of time
def selfConsistency(V, Z):
    tolerance = 1e-6
    if np.abs(V**2 - Z) < tolerance:
        return True
    else:
        return False


#SIAM Hamiltonian - half filled case
def ham(U, V): 
    
    eps_d = -0.5 #energy level of impurity
    eps_k = 1.0 #energy level of bath
    n_bath = 1 #number of bath sites
    
    # Size of the Hamiltonian matrix: impurity (2 states: up/down) + bath states
    dim = 2 + n_bath
    H = np.zeros((dim, dim))
   
    
    
    # Impurity term: diagonal
    H[0, 0] = eps_d  # spin up
    H[1, 1] = eps_d  # spin down
    H[0, 1] = U      # Coulomb interaction (only diagonal term in this basis)
    
    # Bath term: diagonal
    
    H[dim - 1, dim - 1] = eps_k
    
    # Hybridization term
 
    H[0, dim - 1] = V  # spin up
    H[1, dim - 1] = V  # spin down
    H[dim - 1, 0] = V  # Hermitian conjugate
    H[dim - 1, 1] = V  # Hermitian conjugate

    return H


def impGreenFxn(U, V, tvals, t):
    def timeEv(t):
        return expm(-1j * H * t)
    
    
    H = ham(U,V)
    
    # Initial state |psi_0> assumed as the impurity ground state
    dim = H.shape[0]
    psi_0 = np.zeros(dim, dtype=complex)
    psi_0[0] = 1.0  # Impurity spin-up state
    #print("psi: ", psi_0)
    
    # Impurity operator (assuming first state corresponds to impurity)
    d = np.zeros((dim, dim), dtype=complex)
    d[0, 0] = 1.0 # Corresponds to d_sigma annihilation operator

    plus = d.conj().T

    gReal = []

    U_t = timeEv(t)  # e^{-i H t}
    U_t_conj = timeEv(-t) # e^{i H t}
    sub = U_t @ d @ U_t_conj  # Time-evolved operator d(t)



    great = -1j * (psi_0 @ (sub @ plus) @ psi_0)
    less = 1j * (psi_0 @ (plus @ sub) @ psi_0)
    gReal.append(np.real(great-less))

    gReal = np.array(gReal).flatten()
    return gReal



#step 4: using ImpGreenfxn find best fit for params and finding 
def fit_greens_function(tvals, greens_values):
    #print("greens values: ", greens_values)
    def model(x, a, w, p):
        return a * np.cos(w * x) + (1 - a) * np.cos(p * x)

    params, _ = curve_fit(model, tvals, greens_values)
    return params #[a, w, p]

#step 5: calculate quasiparticle weight
def quasiweight(tvals, t, V, U):
    
    a, w, p = fit_greens_function(tvals, impGreenFxn(U, V, tvals, t))
    
    denom = V**4 * ((a/w**4) + ((1-a)/p**4))
    if denom == 0:
        denom = 10

    return 1 / denom

#step 6: put it all together to reach self consistency

Zvals = []

t = np.linspace(1, 50, 500)
V = V_initial

for U in Urange:
    for i in t:
        Z = quasiweight(t, i, U, V)
        
        if selfConsistency(V, Z):
            break
        else:
            V = np.sqrt(Z) 
    Zvals.append(Z)


plt.figure()


plt.plot(Urange, Zvals)

plt.title("Quasiparticle weight Z as a fxn of interaction strength U")
plt.xlabel("U")
plt.ylabel("Z")

plt.show()

### Testing Greens Function ###
# plt.figure()

# xdata = [1,2,3]
# #ydata = [0.84,0.87,0.08]
# ydata = impGreenFxn(U, V, t, np.array([1,0]))
# plt.plot(xdata, ydata, color='green')
# w = 1
# w2 = 1
# a = 0.5
# y = a * np.cos(w*xdata) + (1-a) * np.cos(w2 * xdata)
# plt.plot(xdata,y)


# plt.title("Impurity Greens Function vs Time")
# plt.xlabel("Time")
# plt.ylabel("Impurity Greens Function")

# plt.show()



