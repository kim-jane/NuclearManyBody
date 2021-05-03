from lipkin.hf import HartreeFock
from lipkin.quasi_spin import QuasiSpin
import numpy as np
import matplotlib.pyplot as plt

# plot energy surface vs theta for various chi
epsilon = 1.0
Omega = 10
n = 100
thetas = np.linspace(-np.pi, np.pi, num=n)
chis = [0.5, 1.0, 1.5, 2]
colors = ['r', 'orange', 'green', 'blue']

plt.figure(figsize=(6,4))
for i in range(4):
    
    V = epsilon*chis[i]/(Omega-1)
    HF = HartreeFock(epsilon, V, Omega)
    
    Es = np.zeros(n)
    for j in range(n):
        
        tau = np.array([thetas[j], 0])
        Es[j] = HF.get_energy(tau)
        
    plt.plot(thetas, Es, color=colors[i], label=r'$\chi=$'+str(round(chis[i],1)))
    
    if chis[i] > 1.0:
        
        theta_HF = np.arccos(1.0/chis[i])
        plt.axvline(theta_HF, linestyle='dotted', color=colors[i])
        plt.axvline(-theta_HF, linestyle='dotted', color=colors[i])
  
plt.axvline(-10, linestyle='dotted', color='k', label=r'$\pm \theta_{HF}$')
plt.xlim(-np.pi, np.pi)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$E_{HF}(\theta)$')
plt.title(r'Hartree-Fock Energy Surface for the Lipkin Model ($\epsilon=1$, $\Omega=N=10$)')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig("plot_hf_surface.pdf", format="pdf")

'''
epsilon = 1.0
Omega = 10
chis = [0.9, 1.0, 1.1]
Vs = [epsilon*chi/(Omega-1) for chi in chis]


# collect data for plot
num_points = 1000
thetas = np.linspace(-1, 1, num=num_points)
fig, axs = plt.subplots(1, len(Vs), figsize=(3*len(Vs), 4), sharex=True, sharey=True)

for v in range(len(Vs)):

    HF = HartreeFock(epsilon, Vs[v], Omega)

    Es = np.empty((num_points, 4))
    for i in range(num_points):

        Es[i,0] = HF.solve_equations(num_iter=5, theta0=thetas[i])
        Es[i,1] = HF.solve_equations(num_iter=10, theta0=thetas[i])
        Es[i,2] = HF.solve_equations(num_iter=20, theta0=thetas[i])
        Es[i,3] = HF.solve_equations(num_iter=50, theta0=thetas[i])
        

    axs[v].plot(thetas, Es[:,0], color='r', label='5 iterations')
    axs[v].plot(thetas, Es[:,1], color='orange', label='10 iterations')
    axs[v].plot(thetas, Es[:,2], color='g', label='20 iterations')
    axs[v].plot(thetas, Es[:,3], color='blue', label='50 iterations')
    axs[v].grid(alpha=0.2)
    axs[v].title.set_text(r'$\chi$ = '+str(chis[v]))
    
plt.xlim(-1,1)
axs[0].set(ylabel=r'Hartree-Fock Energy $E_{HF}$')
axs[1].set(xlabel=r'Initial $\theta$')
axs[2].legend()
plt.suptitle(r'Progression to HF Solution for Lipkin Model ($\epsilon=1$, $\Omega=N=10$)')
plt.tight_layout()
plt.savefig("plot_HF_iter.pdf", format="pdf")



'''
