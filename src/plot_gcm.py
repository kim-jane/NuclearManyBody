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
Omega = 10
chis = [0.1, 1, 2, 10]

fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
colors = ['b', 'r', 'g']
states = ['Ground State', '1st Excited State', '2nd Excited State']

for i in range(2):
    for j in range(2):

        V = epsilon*chis[2*i+j]/(Omega-1)
        
        GCM = GeneratorCoordinateMethod(epsilon, V, Omega)
        GCM.display_params()
        GCM.construct_hamiltonian()
        
        thetas, wvfuncs = GCM.get_collective_wavefuncs()
        
        for k in range(3):
            axs[i,j].plot(thetas, wvfuncs[k].real, color=colors[k], label=r'Real Part '+states[k])
            axs[i,j].plot(thetas, wvfuncs[k].imag, color=colors[k], linestyle='dashed', label=r'Imaginary Part '+states[k])
            
        axs[i,j].title.set_text(r'$\chi$ = '+str(chis[2*i+j]))
        axs[i,j].grid(alpha=0.2)

plt.subplots_adjust(left=0.1, right=0.7)
axs[1,1].legend(loc='center left', bbox_to_anchor=(1, 1.2))
axs[1,0].set(xlabel=r'$\theta$')
axs[1,1].set(xlabel=r'$\theta$')
plt.xlim(-np.pi, np.pi)
plt.ylim(-1,1)
plt.suptitle(r'GCM Collective Wave Functions g($\theta$) for Lipkin Model ($\epsilon=1$, $\Omega=N=10$)')
plt.savefig("plot_GCM_wvfuncs.pdf", format="pdf")
plt.clf()
