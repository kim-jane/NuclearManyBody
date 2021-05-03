from pairing.config_int import ConfigurationInteraction
from pairing.bcs import BardeenCooperSchrieffer
import numpy as np
import matplotlib.pyplot as plt

# Plot Energy vs Particle Number

epsilon = 0.0
G = 0.1
Omega = 10
Ns = np.arange(Omega+1)
Es = np.empty(len(Ns))
exact = np.empty(len(Ns))

plt.figure(figsize=(8,6))

for i in range(len(Ns)):

    BCS = BardeenCooperSchrieffer(epsilon, G, Omega, Ns[i])
    u, v, Delta, mu = BCS.solve_equations()
    
    Es[i] = epsilon*Omega*v**2 - 0.25*G*Omega**2*v**2
    
    S = 0.5*Omega
    exact[i] = -G*(S*(S+1) - 0.25*(Ns[i]-Omega)**2 + 0.5*(Ns[i]-Omega))
    

plt.plot(Ns, Es, color='b', label=r'$BCS$')
plt.plot(Ns, exact, color='k', label=r'$Exact$')

plt.xlim(Ns[0], Ns[-1])
plt.xlabel(r'$N$')
plt.ylabel(r'Ground State Energy')
plt.title(r'The BCS Ground State Energy for the Seniority Model ($\epsilon=1$, $G = 0.1$, $\Omega=10$)')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig("plot_bcs_energy.pdf", format="pdf")



# Plot Gap Energy vs Particle Number

epsilon = 0.0
G = 0.1
Omega = 10
Ns = np.arange(Omega+1)
Deltas = np.empty(len(Ns))
plt.figure(figsize=(8,6))

for i in range(len(Ns)):

    BCS = BardeenCooperSchrieffer(epsilon, G, Omega, Ns[i])
    u, v, Deltas[i], mu = BCS.solve_equations()
    
plt.plot(Ns, Deltas, color='b')

plt.xlim(Ns[0], Ns[-1])
plt.xlabel(r'$N$')
plt.ylabel(r'$\Delta$')
plt.title(r'The BCS Gap Energy for the Seniority Model ($\epsilon=1$, $G = 0.1$, $\Omega=10$)')
plt.grid(alpha=0.2)
plt.savefig("plot_bcs_gap.pdf", format="pdf")


    

'''
r = np.linspace(0, 1, num=100)
plt.plot(r, 0.5*G*Omega*np.sqrt(r*(1-r)), color='k', label='Analytic Solution')
plt.scatter(Ns/Omega, Deltas, color='b', label='Iterative Solution')

plt.legend()
plt.grid(alpha=0.2)
plt.savefig("plot_seniority_gap.pdf", format="pdf")
plt.clf()


# Plot Occupation Probability v^2 vs epsilon for various Delta

epsilons = np.linspace(0, 2, num=1000)
G = 1
Omega = 10
N = Omega/2
mu = 1

Deltas = [0.0, 0.1, 0.5, 1.0]
us = np.empty((len(epsilons), len(Deltas)))
vs = np.empty((len(epsilons), len(Deltas)))

for d in range(len(Deltas)):
    for i in range(len(epsilons)):

        BCS = BardeenCooperSchrieffer(epsilons[i], G, Omega, N)
        BCS.display_params()
        
        us[i,d], vs[i,d] = BCS.get_amplitudes(Deltas[d], mu)
    
    
plt.plot(epsilons, np.square(vs[:,0]), color='k')
plt.plot(epsilons, np.square(vs[:,1]), color='b')
plt.plot(epsilons, np.square(vs[:,2]), color='g')
plt.plot(epsilons, np.square(vs[:,3]), color='r')

plt.grid(alpha=0.2)
plt.savefig("plot_seniority_v2.pdf", format="pdf")
plt.clf()
'''
