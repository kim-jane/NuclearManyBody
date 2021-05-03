import numpy as np
from pairing.model import PairingModel


class BardeenCooperSchrieffer(PairingModel):

    name = 'BCS'
    
    def __init__(self, epsilon, G, Omega, N):
        PairingModel.__init__(self, epsilon, G, Omega, N)
        
    
    def solve_equations(self, num_iter=5):
    
        t = 0.1
        u, v = np.sin(t), np.cos(t)
        
        print("\n%10s%10s%10s%10s%10s" % ("Iteration", "u", "v", "Delta", "mu"))
        
        for i in range(num_iter+1):
        
            Delta = self.get_gap(u, v)
            mu = self.get_chemical_potential(Delta)
            
            print("%10i%10.5f%10.5f%10.5f%10.5f" % (i, u, v, Delta, mu))
            
            u, v = self.get_amplitudes(Delta, mu)
            
        return u, v, Delta, mu
                    
        
    def get_gap(self, u, v):
        
        return 0.5*self.Omega*self.G*u*v
        
        
    def get_chemical_potential(self, Delta):
    
        mu = (1-2.0*self.N/self.Omega)*Delta
        mu /= np.sqrt(1-(1-2.0*self.N/self.Omega)**2 + 1.0E-15)
        mu = self.epsilon-mu

        return mu


    def get_amplitudes(self, Delta, mu):
        
        u2 = (self.epsilon-mu)/np.sqrt((self.epsilon-mu)**2+Delta**2)
        v2 = 0.5*(1-u2)
        u2 = 0.5*(1+u2)
        
        return np.sqrt(u2), np.sqrt(v2)
        

    
