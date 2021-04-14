import numpy as np
import itertools
import math
from lipkin.lipkin_model import LipkinModel


class ConfigurationInteraction(LipkinModel):

    basis_name = 'configuration interaction'

    def __init__(self, epsilon, V, Omega, N):

        LipkinModel.__init__(self, epsilon, V, Omega, N)
        self.construct_sp_states()
        self.construct_mb_states()
        
    
    def construct_sp_states(self):
    
        self.num_sp_states = 2*self.Omega
        self.sp_states = []
        for s in [-1, 1]:
            for m in range(self.Omega):
                self.sp_states.append((m, s))
                
                
    def construct_mb_states(self):

        self.mb_states = list(itertools.combinations(np.arange(self.num_sp_states), self.N))
        self.num_mb_states = len(self.mb_states)
        

    def get_sp_index(self, sp_state):
    
        m, s = sp_state
        if s == -1:
            return m
        elif s == 1:
            return self.Omega+m

    def get_connected_configs(self, config):
    
        movable_pairs = []
        connected_configs = []
        two_sp_states = list(itertools.combinations(config, 2))
        
        for sp_index1, sp_index2 in two_sp_states:
        
            m1, s1 = self.sp_states[sp_index1]
            m2, s2 = self.sp_states[sp_index2]
            
            if (m1 != m2) and (s1 == s2):
                
                moved_sp_index1 = self.get_sp_index((m1, -s1))
                
                if moved_sp_index1 not in config:
                
                    moved_sp_index2 = self.get_sp_index((m2, -s2))
                    
                    if moved_sp_index2 not in config:
                            
                        pair = (sp_index1, sp_index2)
                        movable_pairs.append(pair)
                        
                        moved_config = [moved_sp_index1, moved_sp_index2]
                        for sp_index in config:
                            if sp_index not in pair:
                                moved_config.append(sp_index)
                        moved_config.sort()
                        
                        connected_configs.append(tuple(moved_config))
                        
        connected_configs.sort()
        
        return connected_configs

    def construct_hamiltonian(self):
    
        self.H = np.zeros((self.num_mb_states, self.num_mb_states))
        
        # one-body part
        for i in range(self.num_mb_states):
            for sp_index in self.mb_states[i]:
                
                m, s = self.sp_states[sp_index]
                self.H[i,i] += 0.5*self.epsilon*s
        
        # two-body part
        for i in range(self.num_mb_states-1):
            for j in range(i+1, self.num_mb_states):
            
                if self.mb_states[i] in self.get_connected_configs(self.mb_states[j]):
                    self.H[i,j] -= 0.5*self.V
                    
                if self.mb_states[j] in self.get_connected_configs(self.mb_states[i]):
                    self.H[i,j] -= 0.5*self.V
                    
                self.H[j,i] = self.H[i,j]
                
    
        self.eigvals = np.linalg.eigvals(self.H)

    def get_quantum_numbers(self, mb_state):
    
        # K0
        N_minus = 0
        for sp_index in mb_state:
            if sp_index < self.Omega:
                N_minus += 1.0
        K0 = 0.5*self.N - N_minus
        
        # K
        Kminus_Kplus = len(self.get_K('minus', self.get_K('plus', [mb_state])))
        Kplus_Kminus = len(self.get_K('plus', self.get_K('minus', [mb_state])))
        
        K2 = K0**2 + 0.5*(Kplus_Kminus + Kminus_Kplus)
        K = 0.5*(-1+np.sqrt(1+4*K2))
                    
        return K, K0
    

    def get_K(self, mode, mb_states):
    
        K = []
        
        for mb_state in mb_states:
            for sp_index in mb_state:
            
                m, s = self.sp_states[sp_index]
                
                if mode == 'plus' and s == -1:
                
                    new_sp_index = self.get_sp_index((m, -s))
                    
                    if new_sp_index not in mb_state:
                    
                        K_state = list(mb_state)
                        K_state[K_state.index(sp_index)] = new_sp_index
                        K.append(tuple(K_state))
                        
                if mode == 'minus' and s == 1:
                
                    new_sp_index = self.get_sp_index((m, -s))
                    
                    if new_sp_index not in mb_state:
                    
                        K_state = list(mb_state)
                        K_state[K_state.index(sp_index)] = new_sp_index
                        K.append(tuple(K_state))
        
        return K


    def get_state_labels(self):
        
        state_labels = []
    
        for mb_state in self.mb_states:
            
            state_labels.append('s.p. indices = '+str(mb_state))
    
        return state_labels
