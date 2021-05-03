import numpy as np
import scipy.special



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

        propagation = np.zeros((self.num_steps+1, self.dim))
        for i in range(self.num_steps+1):
            
            propagation[i] = psi
            psi = np.dot(propagator, psi)
            psi /= np.linalg.norm(psi)
    
        return propagation



