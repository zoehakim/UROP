import numpy as np
import matplotlib.pyplot as plt
from dmftclass import DMFTroutine


# # Define parameters
# U_range = np.linspace(1,6,6) # Range of U values
# V_init = 1.0 # Initial guess for V

# trial = DMFTroutine(U_range,V_init)

# Z_values = trial.findZ()
# # print(Z_values)

# plt.figure()
# plt.plot(U_range, Z_values, marker="o")
# plt.xlabel("Impurity On-Site Interaction Energy (U)")
# plt.ylabel("Quasiparticle Weight (Z)")
# plt.title("Quasiparticle Weight vs U")
# plt.grid()
# plt.show()

# Define U values and iterate through them
Uvals = np.arange(0.5,6.0,0.5)
V_init = 1.0
trial = DMFTroutine(Uvals,V_init)
Z = []
V = V_init
for U in Uvals:
    print(U)
    V = trial.DMFT_run(U, V, 150)
    Z.append(V**2)

# Plot the results
plt.figure()
plt.plot(Uvals, Z)
plt.xlabel('U')
plt.ylabel('Z')
plt.title('DMFT Results')
plt.show()