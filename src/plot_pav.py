from lipkin.hf import HartreeFock
import numpy as np
import matplotlib.pyplot as plt

epsilon = 1.0
Omega = 10
n = 100
chis = np.linspace(0, 3, num=n)
Ws = np.zeros((2, n))

for i in range(n):
    
    V = epsilon*chis[i]/(Omega-1)
    HF = HartreeFock(epsilon, V, Omega)
    
    if chis[i] < 1.0:
        theta_HF = 0.0
    else:
        theta_HF = np.arccos(1.0/chis[i])
    
    tau = np.array([theta_HF, 0])
    
    Ws[0,i] = HF.get_weight(1.0, tau)
    Ws[1, i] = HF.get_weight(-1.0, tau)
    
plt.plot(chis, Ws[0], color='r', label=r'$r = r_{g.s}$')
plt.plot(chis, Ws[1], color='b', label=r'$r = -r_{g.s}$')

plt.xlim(chis[0], chis[-1])
plt.xlabel(r'$\chi$')
plt.ylabel(r'$W_r(\tau_{HF})$')
plt.title(r'The Signature Weight for the Lipkin Model ($\epsilon=1$, $\Omega=N=10$)')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig("plot_weight.pdf", format="pdf")
plt.show()
