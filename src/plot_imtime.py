from imtime import ImaginaryTime
from lipkin.config_int import ConfigurationInteraction
from lipkin.quasi_spin import QuasiSpin
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

epsilon = 1
V = 0.1
Omega = 2
N = 2
T = 10.0
dt = 0.01
IT = ImaginaryTime(T, dt)
t = np.arange(0, IT.num_steps+1)*dt

# plot imaginary-time propagation in quasi-spin basis
QS = QuasiSpin(epsilon, V, Omega, N)
QS.construct_hamiltonian()
state_labels = QS.get_state_labels()
propagation = IT.get_ground_state(QS.H)

plt.figure(figsize=(10,6))
colors = ['r', 'orange', 'g', 'b']

for i in range(4):
    plt.plot(t, propagation[:,i], lw=4, label=state_labels[i], color=colors[i])

plt.grid(alpha=0.2)
plt.legend()
plt.title('Imaginary-Time Propagation for LM Hamiltonian ($\epsilon=1$, V=0.1, $\Omega=N=2$)')
plt.ylabel('Expansion Coefficients in '+QS.name.title()+' Basis')
plt.xlabel(r'Imaginary Time $\tau$')
plt.xlim(0, T)
plt.savefig("plot_imtime_QS.pdf", format="pdf")
plt.clf()

# plot imaginary-time propagation in quasi-spin basis
CI = ConfigurationInteraction(epsilon, V, Omega, N)
CI.construct_hamiltonian()
state_labels = CI.get_state_labels()
propagation = IT.get_ground_state(CI.H)

plt.figure(figsize=(10,6))
colors = ['r', 'orange', 'gold', 'g', 'b', 'purple']

for i in range(6):
    plt.plot(t, propagation[:,i], lw=4, label=state_labels[i], color=colors[i])

plt.grid(alpha=0.2)
plt.legend()
plt.title('Imaginary-Time Propagation for LM Hamiltonian ($\epsilon=1$, V=0.1, $\Omega=N=2$)')
plt.ylabel('Expansion Coefficients in '+CI.name.title()+' Basis')
plt.xlabel(r'Imaginary Time $\tau$')
plt.xlim(0, T)
plt.savefig("plot_imtime_CI.pdf", format="pdf")
plt.clf()
    
