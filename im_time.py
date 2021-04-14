import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class ImaginaryTime:

    def __init__(self, T, dt):
        
        self.T = T
        self.dt = dt
        self.num_steps = int(np.ceil(T/dt))

    def display_params():
        
        print("IMAGINARY-TIME PROPAGATION")
        print("\tMaximum T = %.3f" % self.T)
        print("\tSpacing dt = %.3f" % self.dt)
        print("\tNumber of steps = %i" % self.num_steps)
        

    def get_ground_state(self, H):
    
        self.dim = H.shape[0]
        psi = np.ones(self.dim)/np.sqrt(self.dim)
        propagator = scipy.linalg.expm(-self.dt*H)

        self.propagation = np.zeros((self.num_steps+1, self.dim))
        for i in range(self.num_steps+1):
            
            self.propagation[i] = psi
            psi = np.dot(propagator, psi)
            psi /= np.linalg.norm(psi)
    
        return psi


    def plot_propagation(self, state_labels, basis_name, file_name):
        
        plt.figure(figsize=(10,6))
        colors = cm.hsv_r(np.linspace(0, 0.9, self.dim))
        
        t = np.arange(0, self.num_steps+1)*self.dt
        for i in range(self.dim):
            plt.plot(t, self.propagation[:,i], lw=4, label=state_labels[i], color=colors[i])
        
        plt.grid(alpha=0.2)
        plt.legend()
        plt.title('Imaginary-Time Propagation')
        plt.ylabel('Expansion Coefficients in '+basis_name.title()+' Basis')
        plt.xlabel(r'Imaginary Time $\tau$')
        plt.xlim(0, self.T)
        plt.savefig(file_name, format='pdf')
