import numpy as np
import tequila as tq

def make_hcb_grouping(H):
    H1 = tq.QubitHamiltonian()
    H2 = tq.QubitHamiltonian()
    H3 = tq.QubitHamiltonian()
    
    U1 = tq.QCircuit()
    U2 = tq.gates.H([i for i in H.qubits])
    U3 = tq.gates.Rx(angle=-np.pi/2, target=[i for i in H.qubits])
    for p in H.paulistrings:
      q = p.naked().qubits
      if p.is_all_z():
          H1 += tq.QubitHamiltonian().from_paulistrings(p)
      else:
          if (p.naked()[q[0]] == "X"):
              for k, v in p.items():
                  p._data[k] = "Z"
              H2 += tq.QubitHamiltonian().from_paulistrings(p)
          else:
              for k, v in p.items():
                  p._data[k] = "Z"
              H3 += tq.QubitHamiltonian().from_paulistrings(p)
    
    hamiltonians = [H1, H2, H3]
    circuits = [U1, U2, U3]
    suggested_samples = [None for _ in range(len(hamiltonians))]

    result = [(H, U) for H, U in zip(hamiltonians, circuits)]
    return result, suggested_samples

