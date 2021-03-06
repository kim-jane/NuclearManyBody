import numpy as np
from lipkin.model import LipkinModel

class HartreeFock(LipkinModel):

    name = 'Hartree-Fock'

    def __init__(self, epsilon, V, Omega):
    
        if Omega%2 == 1:
            raise ValueError('This HF implementation assumes N = Omega = even.')
            
        LipkinModel.__init__(self, epsilon, V, Omega, Omega)
        self.r_gs = (-1)**(0.5*self.Omega)
        self.err = 1E-8
        
    
    def solve_equations(self, num_iter=100, theta0=0.0, phi0=0.0):
    
        # set initial tau
        tau = np.array([theta0, phi0])
        
        # construct HF hamiltonian
        h = self.get_self_consistent_hamiltonian(tau)
        
        # construct kinetic energy
        T = np.zeros((2,2), dtype=np.complex128)
        T[0,0] = -0.5*self.epsilon*self.Omega
        T[1,1] = 0.5*self.epsilon*self.Omega
        
        # container for single particle potential
        Gamma = np.zeros((2,2), dtype=np.complex128)
        
        for i in range(num_iter):
        
            # solve eigenvalue problem
            eigvals, eigvecs = np.linalg.eig(h)
            
            # construct new density matrix
            rho = np.outer(eigvecs[:,0], np.conjugate(eigvecs[:,0]))
            
            # construct new potential
            Gamma[0,1] = -self.V*self.Omega*(self.Omega-1)*rho[1,0]
            Gamma[1,0] = -self.V*self.Omega*(self.Omega-1)*rho[0,1]
            
            # construct new hamiltonian
            h = T + Gamma
            
            # calculate energy
            E = 0.5*np.trace(np.dot(T+h, rho)).real
            
        return E
        
        
    def get_self_consistent_hamiltonian(self, tau):
        
        theta, phi = tau[0], tau[1]
        h = np.empty((2,2), dtype=np.complex128)
        
        h[0,0] = 1
        h[1,1] = -1
        h[0,1] = self.chi*np.sin(theta)*np.exp(1j*phi)
        h[1,0] = self.chi*np.sin(theta)*np.exp(-1j*phi)
        
        return -0.5*self.epsilon*self.Omega*h
        
        
    def minimize_energy(self, num_iter=10000):
    
        # pick small initial tau = (theta, phi)
        tau = np.random.normal(0.0, 0.1, 2)
        
        # initialize adam optimizer
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        
        # start minimizing
        for self.t in range(1, num_iter+1):
    
            E = self.get_energy(tau)
            grad = self.get_gradient_energy(tau)
            tau = self.update_tau(tau, grad)
        
        return tau
        
        
    def minimize_signature_projected_energy(self, r, num_iter=10000):
    
        # pick small initial tau = (theta, phi)
        tau = np.random.normal(0.0, 0.1, 2)
        
        # initialize adam optimizer
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        
        # start minimizing
        for self.t in range(1, num_iter+1):
    
            Er = self.get_signature_projected_energy(r, tau)
            grad = self.get_gradient_projected_energy(r, tau)
            tau = self.update_tau(tau, grad)

        return tau
        
    def get_energy(self, tau):
    
        theta, phi = tau[0], tau[1]
        E = np.cos(theta)+0.5*self.chi*(np.sin(theta)**2)*np.cos(2*phi);
        
        return -0.5*self.epsilon*self.Omega*E
        
        
    def get_gradient_energy(self, tau):
    
        theta, phi = tau[0], tau[1]
        factor = 0.5*self.epsilon*self.Omega*np.sin(theta)
        dE_dtheta = factor*(1-self.chi*np.cos(theta)*np.cos(2*phi))
        dE_dphi = factor*self.chi*np.sin(theta)*np.sin(2*phi)

        return np.array([dE_dtheta, dE_dphi])
        
    def get_weight(self, r, tau):
    
        theta = tau[0]
        a = 1.0+r*self.r_gs*(np.cos(theta))**(self.Omega-2)
        b = 1.0+r*self.r_gs*(np.cos(theta))**self.Omega
        
        if a < self.err and b < self.err:
            return float((self.Omega-2))/float(self.Omega)
        
        else:
            return (a+self.err)/(b+self.err)
    
    def get_gradient_weight(self, r, tau):
    
        theta = tau[0]
        a = 2*(1+r*self.r_gs*(np.cos(theta))**self.Omega)-self.Omega*(np.sin(theta))**2
        a *= r*self.r_gs*np.sin(theta)*(np.cos(theta))**(self.Omega-3)
        b = (1+r*self.r_gs*(np.cos(theta))**self.Omega)**2
        
        if a < self.err and b < self.err:
            return np.array([theta*float((self.Omega-2))/float(self.Omega), 0])
        
        return np.array([(a+self.err)/(b+self.err), 0])
        

    def get_signature_projected_energy(self, r, tau):

        return self.get_energy(tau)*self.get_weight(r, tau)
        
        
    def get_gradient_projected_energy(self, r, tau):
    
        E = self.get_energy(tau)
        W = self.get_weight(r, tau)
        gradE = self.get_gradient_energy(tau)
        gradW = self.get_gradient_weight(r, tau)
        
        return E*gradW + W*gradE
        

    def update_tau(self, tau, gradient, eta0=0.001, beta1=0.9, beta2=0.999, epsilon=1.0E-8):
    
        eta = eta0*np.sqrt(1.0-beta2**self.t)/(1.0-beta1**self.t)
        self.m = beta1*self.m+(1.0-beta1)*gradient;
        self.v = beta2*self.v+(1.0-beta2)*np.square(gradient);
        tau -= eta*np.divide(self.m, np.sqrt(self.v)+epsilon)
        self.t += 1
        
        return tau
        


