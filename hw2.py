from lipkin.quasi_spin import QuasiSpin
from lipkin.config_int import ConfigurationInteraction
from tools.display import display_hamiltonian, display_eigvals, display_sp_states, display_mb_states


# eigenvalues of LM hamiltonian for large half-filled system
epsilon = 1.0
V = 0.1
Omega = 10
N = 10

QS = QuasiSpin(epsilon, V, Omega, N)
QS.display_params()
QS.construct_hamiltonian()

display_mb_states(QS.mb_states, ['K', 'K0', 'r'])
display_eigvals(QS.eigvals)

# comparing eigenvalues of LM hamiltonian for small half-filled system
Omega = 2
N = 2

QS = QuasiSpin(epsilon, V, Omega, N)
QS.display_params()
QS.construct_hamiltonian()

display_mb_states(QS.mb_states, ['K', 'K0', 'r'])
display_hamiltonian(QS.H)
display_eigvals(QS.eigvals)

CI = ConfigurationInteraction(epsilon, V, Omega, N)
CI.display_params()
CI.construct_hamiltonian()

display_sp_states(CI.sp_states, ['m', 's'])
display_mb_states(CI.mb_states, ['sp_index'+str(i) for i in range(N)])
display_hamiltonian(CI.H)
display_eigvals(CI.eigvals)

# comparing eigenvalues of LM hamiltonian for small filled system
Omega = 2
N = 4

QS = QuasiSpin(epsilon, V, Omega, N)
QS.display_params()
QS.construct_hamiltonian()

display_hamiltonian(QS.H)
display_eigvals(QS.eigvals)

CI = ConfigurationInteraction(epsilon, V, Omega, N)
CI.display_params()
CI.construct_hamiltonian()

display_hamiltonian(CI.H)
display_eigvals(CI.eigvals)