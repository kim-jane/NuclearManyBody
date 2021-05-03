from lipkin.gcm import GeneratorCoordinateMethod
from lipkin.quasi_spin import QuasiSpin
from tools.display import display_eigvals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# solve GCM eigenvalue problem for N=Omega=10
# compare to exact solutions
# plot collective wave functions
epsilon = 1.0
V = 10
Omega = 10


GCM = GeneratorCoordinateMethod(epsilon, V, Omega)
GCM.display_params()
GCM.construct_hamiltonian()

GCM.plot_collective_wavefuncs('plot_GCM_wvfuncs_V10.pdf')
GCM.plot_collective_wavefuncs('plot_GCM_wvfuncs_polar_V10.pdf', polar=True)
GCM.plot_collective_wavefuncs('plot_GCM_wvfuncs_square_V10.pdf', square=True)
GCM.plot_collective_wavefuncs('plot_GCM_wvfuncs_square_polar_V10.pdf', square=True, polar=True)



'''
print("\nGCM ", end='')
display_eigvals(GCM.eigvals)

QS = QuasiSpin(epsilon, V, Omega, Omega)
QS.display_params()
QS.construct_hamiltonian()

print("\nExact ", end='')
display_eigvals(np.sort(QS.eigvals))


# collect data for plot
V = np.linspace(0, 0.5, num=10)
chi = []
E_GCM = np.zeros((len(V), len(GCM.eigvals)))
E_exact = np.zeros((len(V), len(QS.eigvals)))

for i in range(len(V)):

    GCM = GeneratorCoordinateMethod(epsilon, V[i], Omega)
    GCM.construct_hamiltonian()
    E_GCM[i] = GCM.eigvals
    
    QS = QuasiSpin(epsilon, V[i], Omega, Omega)
    QS.construct_hamiltonian()
    E_exact[i] = QS.eigvals
    chi.append(QS.chi)


# plot GCM and exact ground state and first excited state energies
fig, ax = plt.subplots()

ax.plot(chi, E_exact[:,0], color='b', lw=2, label='Exact Ground State Energy')
ax.plot(chi, E_GCM[:,0], color='r', marker='o', linestyle='dashed', label='GCM Ground State Energy')


plt.title(r'Ground State Energy of LM Hamiltonian ($\epsilon=1$, $N = \Omega=10$)')
plt.xlabel(r'$\chi$')
plt.ylabel(r'Energy')
plt.xlim(chi[0], chi[-1])
plt.grid(alpha=0.2)
plt.legend()
plt.savefig('plot_GCM_gs_energy.pdf', format='pdf')
plt.clf()

# plot all eigenvalues
fig, ax = plt.subplots()

for i in range(len(QS.eigvals)):
    ax.plot(chi, E_exact[:, i], linewidth=0.5, alpha=0.7, color='k')
    
ax.plot(chi, -100*np.ones(len(chi)), linewidth=0.5, alpha=0.7, color='k', label='Exact')

for i in range(len(GCM.eigvals)):
    ax.scatter(chi, E_GCM[:, i], marker='o', color='r', s=2)

ax.scatter(chi, -100*np.ones(len(chi)), color='r', marker='o', s=2, label='GCM')

plt.title('Energy Spectrum of LM Hamiltonian ($\epsilon=1$, $N = \Omega=10$)')
plt.xlabel(r'$\chi$')
plt.ylabel(r'Energy')
plt.xlim(chi[0], chi[-1])
plt.ylim(E_GCM.min()-0.1, E_GCM.max()+0.1)
plt.grid(alpha=0.2)
plt.legend()
plt.savefig('plot_GCM_all_energies.pdf', format='pdf')
'''
