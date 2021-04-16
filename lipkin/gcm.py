import numpy as np
from lipkin.lipkin_model import LipkinModel
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GeneratorCoordinateMethod(LipkinModel):

    basis_name = "GCM"

    def __init__(self, epsilon, V, Omega):
    
        if Omega%2 == 1:
            raise ValueError('This GCM implementation assumes N = Omega = even.')
            
        LipkinModel.__init__(self, epsilon, V, Omega, Omega)
        
        
    def construct_hamiltonian(self, num_points=100):
    
        dim = self.Omega+1
        dtheta = 2*np.pi/(num_points-1)
        
        self.possible_k = [-0.5*self.Omega+i for i in range(dim)]
        theta = np.linspace(-np.pi, np.pi, num=num_points)
    
        # construct hamiltonian
        self.H = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            for j in range(dim):
            
                k1 = self.possible_k[i]
                k2 = self.possible_k[j]
            
                # precalculate factors that do not depend on theta
                factor = (2**(self.Omega-2))*((dtheta)**2)/(np.pi**2)
                factor /= np.sqrt(scipy.special.binom(self.Omega, 0.5*self.Omega+k1))
                factor /= np.sqrt(scipy.special.binom(self.Omega, 0.5*self.Omega+k2))
                
                # simple numerical integration
                for t1 in theta:
                    for t2 in theta:
                        self.H[i, j] += factor*np.exp(1j*(k1*t1-k2*t2))*self.calc_H(t1, t2)
        
        self.eigvals, self.eigvecs = np.linalg.eig(self.H)
        print(self.eigvals)
        self.eigvals = self.eigvals.real
        idx = self.eigvals.argsort()
        self.eigvals = self.eigvals[idx]
        self.eigvecs = self.eigvecs[:,idx]
        
        
    def plot_collective_wavefuncs(self, filename, num_lowest_states=2, square=False, polar=False):
    
        num_points = 100
        t = np.array([-np.pi+i*(2*np.pi/num_points) for i in range(num_points+1)])
        colors = ['b', 'r', 'g']
        states = ['Ground State', '1st Excited State', '2nd Excited State']
        if num_lowest_states > 3:
            print('Add more colors and state labels. ')
    
        if polar:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        else:
            fig, ax = plt.subplots()
            plt.grid(alpha=0.2)
        

        for i in range(num_lowest_states):
            
            g = self.eigvecs[i]
            E = self.eigvals[i]
            
            collective = np.zeros(num_points+1, dtype=np.complex128)
            
            for j in range(self.Omega+1):
                
                k = self.possible_k[j]
                collective += g[j]*np.exp(-1j*k*t)
                
            collective /= np.sqrt(2*np.pi)
            
            if square:
                ax.plot(t, np.square(collective).real, color=colors[i], label=states[i])
            else:
                ax.plot(t, collective.real, color=colors[i], label=r'Real Part '+states[i])
                ax.plot(t, collective.imag, color=colors[i], linestyle='dashed', label='Imaginary Part '+states[i])
        
        if square:
            title = r'Square of Collective Wave Functions $|g(\theta)|^2$'
        else:
            title = r'Collective Wave Functions g($\theta$)'
        title += ' for Lowest '+str(num_lowest_states)+' States'
        plt.title(title)
        plt.xlabel(r'$\theta$')
        plt.xlim(-np.pi, np.pi)
        
        if polar:
            plt.yticks([-1,0,1], [-1, 0, 1])
            plt.legend(bbox_to_anchor=(0.5, 0.2), loc='upper left')
        else:
            plt.legend()
        plt.savefig(filename, format='pdf')
            


    def calc_H(self, theta1, theta2):
        
        H = (1+(np.sin(0.5*(theta1+theta2)))**2)/(np.cos(0.5*(theta1-theta2)))**2
        H = 0.5*self.chi*(H - 1)
        H = np.cos(0.5*(theta1+theta2))/np.cos(0.5*(theta1-theta2)) + H
        
        return -0.5*self.epsilon*self.Omega*H*(np.cos(0.5*(theta1-theta2)))**self.Omega




    def calc_n(self, k):
    
        return scipy.special.binom(self.Omega, 0.5*self.Omega+k)*np.pi/2**(self.Omega-1)

