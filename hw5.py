from lipkin.hartree_fock import HartreeFock
from lipkin.quasi_spin import QuasiSpin
import numpy as np
import matplotlib.pyplot as plt


epsilon = 1.0
V = 0.1
Omega = 10
HF = HartreeFock(epsilon, V, Omega)

# PAV - Projection After Variation
tau_HF = HF.minimize_energy()
Er0 = HF.get_signature_projected_energy(HF.r_gs, tau_HF)
Er1 = HF.get_signature_projected_energy(-HF.r_gs, tau_HF)
print("PAV ground state energy = %f" % Er0)
print("PAV first excited state energy = %f" % Er1)

# VAP - Variation After Projection
tau_r0 = HF.minimize_signature_projected_energy(HF.r_gs)
tau_r1 = HF.minimize_signature_projected_energy(-HF.r_gs)
Er0 = HF.get_signature_projected_energy(HF.r_gs, tau_r0)
Er1 = HF.get_signature_projected_energy(-HF.r_gs, tau_r1)
print("VAP ground state energy = %f" % Er0)
print("VAP first excited state energy = %f" % Er1)


# collect data for plot
V = np.linspace(0.0, 0.334, num=100)
n = len(V)

chi = np.empty(n)
E_HF = np.empty(n)
E_PAV = np.empty((n, 2))
E_VAP = np.empty((n, 2))
E_exact = np.empty((n, 2))

for i in range(n):

    HF = HartreeFock(epsilon, V[i], Omega)
    tau_HF = HF.minimize_energy()
    tau_r0 = HF.minimize_signature_projected_energy(HF.r_gs)
    tau_r1 = HF.minimize_signature_projected_energy(-HF.r_gs)
    
    # chi
    chi[i] = HF.chi
    
    # HF energy
    E_HF[i] = HF.get_energy(tau_HF)
    
    # PAV energies
    E_PAV[i,0] = HF.get_signature_projected_energy(HF.r_gs, tau_HF)
    E_PAV[i,1] = HF.get_signature_projected_energy(-HF.r_gs, tau_HF)
    
    # VAP energies
    E_VAP[i,0] = HF.get_signature_projected_energy(HF.r_gs, tau_r0)
    E_VAP[i,1] = HF.get_signature_projected_energy(-HF.r_gs, tau_r1)
    
    # exact
    QS = QuasiSpin(epsilon, V[i], Omega, Omega)
    QS.construct_hamiltonian()
    E_exact[i] = np.sort(QS.eigvals)[:2]

# PLOT - exact, HF, PAV, VAP energies
plt.figure(figsize=(8,6))

ls = ['solid', 'dotted']
lw = 3
plt.plot(chi, E_HF, c='k', linewidth=lw, label='HF')
plt.plot(chi, E_PAV[:,0], c='g', linestyle=ls[0], linewidth=lw, label='PAV ground state')
plt.plot(chi, E_PAV[:,1], c='g', linestyle=ls[1], linewidth=lw, label='PAV 1st excited state')
plt.plot(chi, E_VAP[:,0], c='b', linestyle=ls[0], linewidth=lw, label='VAP ground state')
plt.plot(chi, E_VAP[:,1], c='b', linestyle=ls[1], linewidth=lw, label='VAP 1st excited state')
plt.plot(chi, E_exact[:,0], c='r', linestyle=ls[0], linewidth=lw, label='Exact ground state')
plt.plot(chi, E_exact[:,1], c='r', linestyle=ls[1], linewidth=lw, label='Exact 1st excited state')

plt.grid(alpha=0.2)
plt.legend()
plt.xlim(chi[0], chi[-1])
plt.xlabel(r"$\chi$")
plt.ylabel(r"Energy")
plt.title(r"Comparison of Methods for LM Hamiltonian ($\epsilon$=1, $\Omega$ = N = 10)")
plt.savefig("plot_HF_PAV_VAP.pdf", format='pdf')

