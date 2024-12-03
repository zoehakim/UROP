import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import expm

# Simulate two-site DMFT without quantum circuit

#step 1: set val of impurity on site interaction energy
U = 10.0

#step 2: make an intial gues for the val of hybridization parameter
V = 1.0 

#step 3: obtain impurity Green's function from QComp as a fxn of time

h = 1.0 # planck
c = 3.0 * 10**8 # speed of light

def selfConsistency(V, Z, t):
    if(V**2 == Z * t**2):
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
    
    H[dim - 1,dim - 1] = eps_k
    
    # Hybridization term
 
    H[0, 1 + n_bath] = V  # spin up
    H[1, 1 + n_bath] = V  # spin down
    H[1 + n_bath, 0] = V  # Hermitian conjugate
    H[1 + n_bath, 1] = V  # Hermitian conjugate

    return H

def impGreenFxn(U, V, tvals, t):
    def timeEv(U, V, t):
        return np.exp(-1j * ham(U,V) * t)
    
    H = ham(U,V)

    if t == 0:
        return 1.0
    else:
        # Initial state |psi_0> assumed as the impurity ground state
        dim = H.shape[0]
        psi_0 = np.zeros(dim, dtype=complex)
        psi_0[0] = 1.0  # Impurity spin-up state
        
        # Impurity operator (assuming first state corresponds to impurity)
        d = np.zeros((dim, dim), dtype=complex)
        d[0, 0] = 1.0  # Corresponds to d_sigma annihilation operator
        
        # Compute the real part of G_d(t)
        G_d_t_real = []
        for t in tvals:
            U_t = timeEv(U, V, t)  # e^{-i H t}
            U_t_conj = timeEv(U,V, -t)  # e^{i H t}
            
            d_t = U_t @ d @ U_t_conj  # Time-evolved operator d(t)
            G_t = -1j * (psi_0.conj().T @ d_t @ d @ psi_0).item()  # Expectation value
            G_d_t_real.append(np.real(G_t))  # Extract the real part

        return np.array(G_d_t_real)


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

t = np.linspace(0, 10, 11, True)
tstar = t * np.sqrt(t)



for i in tstar:
    Z = quasiweight(t, i, U, V)
    Zvals.append(Z)
    
    if selfConsistency(V, Z, i):
        break
    else:
        V = np.sqrt(Z) * i 

Urange = U / tstar

# what it should look like:
    # Zvals = [1.0, 0.9, 0.5, 0.0]
    # Urange = [0.0, 2.0, 4.0, 6.0]
    
plt.figure()

plt.plot(Urange, Zvals)

plt.title("Quasiparticle weight Z as a fxn of interaction strength U")
plt.xlabel("U/t*")
plt.ylabel("Z")

plt.show()





