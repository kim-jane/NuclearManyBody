from lipkin.hf import HartreeFock
from lipkin.quasi_spin import QuasiSpin
from lipkin.gcm import GeneratorCoordinateMethod
import numpy as np
import matplotlib.pyplot as plt



# plot lowest 2 energies for all methods
epsilon = 1.0
Omega = 10
n = 50
chi = np.linspace(0.0, 3.0, num=n)

E_exact = np.empty((n, 2))
E_HF = np.empty(n)
E_PAV = np.empty((n, 2))
E_VAP = np.empty((n, 2))
E_GCM = np.empty((n, 2))

for i in range(n):
    
    print(i)
    V = epsilon*chi[i]/(Omega-1)

    # exact
    QS = QuasiSpin(epsilon, V, Omega, Omega)
    QS.construct_hamiltonian()
    E_exact[i] = np.sort(QS.eigvals)[:2]

    # HF methods
    HF = HartreeFock(epsilon, V, Omega)
    tau_HF = HF.minimize_energy(num_iter=5000)
    tau_r0 = HF.minimize_signature_projected_energy(HF.r_gs, num_iter=5000)
    tau_r1 = HF.minimize_signature_projected_energy(-HF.r_gs, num_iter=5000)
    
    # HF energy
    E_HF[i] = HF.get_energy(tau_HF)
    
    # PAV energies
    E_PAV[i,0] = HF.get_signature_projected_energy(HF.r_gs, tau_HF)
    E_PAV[i,1] = HF.get_signature_projected_energy(-HF.r_gs, tau_HF)
    
    # VAP energies
    E_VAP[i,0] = HF.get_signature_projected_energy(HF.r_gs, tau_r0)
    E_VAP[i,1] = HF.get_signature_projected_energy(-HF.r_gs, tau_r1)
    
    # GCM
    GCM = GeneratorCoordinateMethod(epsilon, V, Omega)
    GCM.construct_hamiltonian(num_points=200)
    E_GCM[i] = np.sort(GCM.eigvals)[:2]
    

plt.figure(figsize=(8,6))

ls = ['solid', 'dotted']
lw = 2
plt.plot(chi, E_HF, c='r', linewidth=lw, label='HF')
plt.plot(chi, E_PAV[:,0], c='orange', linestyle=ls[0], linewidth=lw, label='PAV Ground State')
plt.plot(chi, E_PAV[:,1], c='orange', linestyle=ls[1], linewidth=lw, label='PAV 1st Excited State')
plt.plot(chi, E_VAP[:,0], c='g', linestyle=ls[0], linewidth=lw, label='VAP Ground State')
plt.plot(chi, E_VAP[:,1], c='g', linestyle=ls[1], linewidth=lw, label='VAP 1st Excited State')
plt.plot(chi, E_GCM[:,0], c='b', linestyle=ls[0], linewidth=lw, label='GCM Ground State')
plt.plot(chi, E_GCM[:,1], c='b', linestyle=ls[1], linewidth=lw, label='GCM 1st Excited State')
plt.plot(chi, E_exact[:,0], c='k', linestyle=ls[0], linewidth=lw-1, label='Exact Ground State')
plt.plot(chi, E_exact[:,1], c='k', linestyle=ls[1], linewidth=lw-1, label='Exact 1st Excited State')

plt.grid(alpha=0.2)
plt.legend()
plt.xlim(chi[0], chi[-1])
plt.xlabel(r"$\chi$")
plt.ylabel(r"Energy")
plt.title(r"Comparison of Methods for LM Hamiltonian ($\epsilon$=1, $\Omega$ = N = 10)")
plt.savefig("plot_lipkin_energies.pdf", format='pdf')

