import numpy as np
import matplotlib.pyplot as plt
from dmftclass import DMFTroutine


# Define parameters
U_range = np.linspace(1,10,10) # Range of U values
V_init = 10.0  # Initial guess for V

trial = DMFTroutine(U_range,V_init)
Z_values = trial.dmft_routine()

plt.plot(U_range, Z_values, marker="o")
plt.xlabel("Impurity On-Site Interaction Energy (U)")
plt.ylabel("Quasiparticle Weight (Z)")
plt.title("Quasiparticle Weight vs U")
plt.grid()
plt.show()