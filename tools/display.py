import pandas as pd

def display_hamiltonian(H):

    print("Hamiltonian:")
    for i in range(H.shape[0]):
        print("[", end="")
        for j in range(H.shape[1]):
            print("{: 10.5f}".format(H[i,j]), end="")
        print("]")
    print("\n")


def display_eigvals(eigvals):
    
    print("Eigenvalues:")
    for eigval in eigvals:
        print("{: >15.8f}".format(eigval))
    print("\n")

def display_sp_states(sp_states, labels=None):

    print("Single-particle states:")
    print(pd.DataFrame(sp_states, columns=labels))
    print("\n")

def display_mb_states(mb_states, labels=None):

    print("Many-body states:")
    print(pd.DataFrame(mb_states, columns=labels))
    print("\n")
