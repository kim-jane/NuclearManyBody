import numpy as np
from lipkin.model import LipkinModel
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tools.display import display_hamiltonian

class GeneratorCoordinateMethod(LipkinModel):

    name = 'GCM'

    def __init__(self, epsilon, V, Omega):
    
        if Omega%2 == 1:
            raise ValueError('This GCM implementation assumes N = Omega = even.')
            
        LipkinModel.__init__(self, epsilon, V, Omega, Omega)
        
        
    def construct_hamiltonian(self, num_points=500):
    
        dim = self.Omega+1
        dtheta = 2*np.pi/(num_points-1)
        
        self.possible_k = np.array([-0.5*self.Omega+i for i in range(dim)])
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
        self.eigvals = self.eigvals.real
        idx = self.eigvals.argsort()
        self.eigvals = self.eigvals[idx]
        self.eigvecs = self.eigvecs[:,idx]
        
        
    def get_collective_wavefuncs(self, num_points=200, num_lowest_states=3):
    
        thetas = np.linspace(-np.pi, np.pi, num=num_points)
        wvfuncs = np.zeros((num_lowest_states, num_points), dtype=np.complex128)
        
        for i in range(num_lowest_states):
            
            g = self.eigvecs[:, i]
            E = self.eigvals[i]
            
            for j in range(self.Omega+1):
                
                k = self.possible_k[j]
                wvfuncs[i] += g[j]*np.exp(-1j*k*thetas)
                
            wvfuncs[i] /= np.sqrt(2*np.pi)
            
        return thetas, wvfuncs
        


    def calc_H(self, theta1, theta2):
    
        err = 1E-8
        
        H = (1+(np.sin(0.5*(theta1+theta2)))**2)/((np.cos(0.5*(theta1-theta2)))**2+err)
        H = 0.5*self.chi*(H - 1)
        H += np.cos(0.5*(theta1+theta2))/(np.cos(0.5*(theta1-theta2))+err)
        
        return -0.5*self.epsilon*self.Omega*H*(np.cos(0.5*(theta1-theta2)))**self.Omega




    def calc_n(self, k):
    
        return scipy.special.binom(self.Omega, 0.5*self.Omega+k)*np.pi/2**(self.Omega-1)

