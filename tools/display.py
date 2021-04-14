import pandas as pd

def display_hamiltonian(H):

    print("\nHamiltonian:")
    for i in range(H.shape[0]):
        print("[", end="")
        for j in range(H.shape[1]):
            print("{: 10.5f}".format(H[i,j]), end="")
        print("]")


def display_eigvals(eigvals):
    
    print("\nEigenvalues:")
    for eigval in eigvals:
        print("{: >15.8f}".format(eigval))


def display_sp_states(sp_states, labels=None):

    print("\nSingle-particle states:")
    print(pd.DataFrame(sp_states, columns=labels))


def display_mb_states(mb_states, labels=None):

    print("\nMany-body states:")
    print(pd.DataFrame(mb_states, columns=labels))
