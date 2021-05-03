import numpy as np
import itertools
import math
from pairing.model import PairingModel


class ConfigurationInteraction(PairingModel):

    name = 'Configuration Interaction'
    
    def __init__(self, epsilon, G, j, N):
    
        PairingModel.__init__(self, epsilon, G, j, N)
        self.construct_sp_states()
        self.construct_mb_states()
        
    def construct_sp_states(self):
    
        self.num_sp_states = self.Omega
        self.sp_states = np.arange(self.Omega)-self.j
        self.positive_m = np.arange(self.j, -0.1, -1)
    
    def construct_mb_states(self):
        
        self.mb_states = list(itertools.combinations(np.arange(self.num_sp_states), self.N))
        self.num_mb_states = len(self.mb_states)
        
    def get_num_pairs(self, mb_state):
        
        num_pairs = 0
        for m in self.positive_m:
            for sp_index1 in mb_state:
                if self.sp_states[sp_index1] == m:
                    for sp_index2 in mb_state:
                        if self.sp_states[sp_index2] == -m:
                            num_pairs += 1
        return num_pairs
        

    def construct_hamiltonian(self):
    
        self.H = np.zeros((self.num_mb_states, self.num_mb_states))
        
        # one-body + diagonal two-body part
        for i in range(self.num_mb_states):
            
            self.H[i,i] = self.epsilon*self.N
            self.H[i,i] += -self.G*self.get_num_pairs(self.mb_states[i])
            
        # off-diagonal two-body part
        for i in range(self.num_mb_states-1):
        
            n1 = self.get_num_pairs(self.mb_states[i])
            
            for j in range(i+1, self.num_mb_states):
            
                n2 = self.get_num_pairs(self.mb_states[j])
                
                self.H[i,j] = -self.G*n1*n2
                self.H[j,i] = self.H[i,j]
            
        print(self.H)
        print(np.linalg.eigvals(self.H))

