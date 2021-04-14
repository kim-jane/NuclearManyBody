import numpy as np
from lipkin.lipkin_model import LipkinModel


class QuasiSpin(LipkinModel):

    basis_name = 'quasi-spin'

    def __init__(self, epsilon, V, Omega, N):
    
        LipkinModel.__init__(self, epsilon, V, Omega, N)
        self.construct_mb_states()
        
        
    def construct_mb_states(self):

        # determine possible K
        if self.N <= self.Omega:
            self.max_K = 0.5*self.N
        elif self.N <= 2*self.Omega:
            self.max_K = self.Omega - 0.5*self.N
        else:
            raise ValueError("Too many particles!")
        self.possible_K = np.arange(self.max_K, -0.1, -1)

        # construct all many-body states
        self.blocks = []
        self.mb_states = []
        for K in self.possible_K:
        
            # split possible K0 by signature
            r1 = self.get_signature(K)
            r2 = self.get_signature(K-1)
            possible_K0_r1 = np.arange(K, -K-0.1, -2)
            possible_K0_r2 = np.arange(K-1, -K-0.1, -2)
            
            self.blocks.append((K, r1))
            if len(possible_K0_r2) > 0:
                self.blocks.append((K, r2))
            
            for K0 in possible_K0_r1:
                self.mb_states.append((K, K0, r1))
            
            for K0 in possible_K0_r2:
                self.mb_states.append((K, K0, r2))

        self.num_mb_states = len(self.mb_states)
        
        
    def construct_hamiltonian(self):
        
        self.H = np.zeros((self.num_mb_states, self.num_mb_states))
        self.eigvals = []
        
        for (K, r) in self.blocks:
            self.construct_block(K, r)

        
    def construct_block(self, K, r):

        k = self.get_block_index(K, r)
        N = self.get_block_size(K, r)
        block = np.zeros((N, N))

        # diagonal elems
        for i in range(N):
            block[i,i] = self.epsilon * self.mb_states[k+i][1]

        # off-diagonal elems
        for i in range(N-1):
        
            j = i+1
            K0i = self.mb_states[k+i][1]
            K0j = self.mb_states[k+j][1]
            
            if K0i-K0j == 2:
                block[i,j] = np.sqrt(K*(K+1)-K0i*(K0i-1))
                block[i,j] *= np.sqrt(K*(K+1)-(K0i-1)*(K0i-2))
                block[i,j] *= -0.5*self.V
                block[j,i] = block[i,j]
        
        self.H[k:k+N, k:k+N] = block
        self.eigvals += list(np.sort(np.linalg.eigvals(block)))
        
        
    def get_block_index(self, K, r):

        if (K, K, r) in self.mb_states:
            return self.mb_states.index((K, K, r))
        else:
            return self.mb_states.index((K, K-1, r))



    def get_block_size(self, K, r):

        if self.N%2 == 0:
            if r == self.get_signature(K):
                return int(K+1)
            else:
                return int(K)
        else:
            return int(K+0.5)
        
        
    def get_signature(self, K0):
        
        if self.N%2 == 0:
            if K0%2 == 0:
                return 1
            else:
                return -1
        else:
            r = np.exp(1j*np.pi*K0)
            return r.imag*1j
            

    def get_state_labels(self):
    
        state_labels = []
        
        for mb_state in self.mb_states:
            
            state_labels.append('(K, K0, r)='+str(mb_state))

        return state_labels
