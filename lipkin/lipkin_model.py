class LipkinModel:

    def __init__(self, epsilon, V, Omega, N):

        self.epsilon = epsilon
        self.V = V
        self.Omega = Omega
        self.N = N
        self.chi = V*(Omega-1)/epsilon
        

    def display_params(self):
        
        line = "="*60
        print(line)
        
        print("LIPKIN MODEL")
        print("\tEnergy Spacing epsilon = %.3f" % self.epsilon)
        print("\tInteraction Coupling V = %.3f" % self.V)
        print("\tNumber of doublets Omega = %i" % self.Omega)
        print("\tNumber of particles N = %i" % self.N)
        print("\tBasis: %s" % self.basis_name)

