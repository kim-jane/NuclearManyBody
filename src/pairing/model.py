class PairingModel:

    def __init__(self, epsilon, G, Omega, N):

        self.epsilon = epsilon
        self.G = G
        self.Omega = Omega
        self.N = N
        self.j = 0.5*(Omega-1.0)
        
        

    def display_params(self):
        
        line = "="*60
        print(line)
        
        if self.Omega%2 == 0:
            j_str = "2j = "+str(int(2*self.j))
        else:
            j_str = "j = "+str(int(self.j))
        
        print("PAIRING MODEL")
        print("\tSingle %s shell energy epsilon = %.3f" % (j_str, self.epsilon))
        print("\tInteraction coupling G = %.3f" % self.G)
        print("\tNumber of m-substates Omega = %i" % self.Omega)
        print("\tNumber of particles N = %i" % self.N)
        print("\tMethod: %s" % self.name)


