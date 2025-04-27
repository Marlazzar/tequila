import queue
from mqp.qiskit_provider import MQPProvider, MQPBackend
from qiskit.circuit import QuantumCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit, TequilaQiskitException
import qiskit
from calendar import c
from tequila.utils.keymap import KeyMapRegisterToSubregister
from tequila import BitString, BitNumbering, BitStringLSB
from tequila.circuit.compiler import change_basis
import numbers, typing, numpy, copy, warnings
from tequila.objective.objective import Variable, format_variable_dictionary
import typing
from tequila.utils.misc import to_float
import numpy
from typing import Union
from mqp.qiskit_provider import MQPProvider, MQPBackend
from tequila import TequilaException, TequilaWarning, circuit
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
from tequila.circuit import gates 
from tequila.circuit import QCircuit
import numpy as np

from tequila.simulators.simulator_aqt import BackendCircuitAQT, BackendExpectationValueAQT


# TODO: mit aqt mergen

class BackendCircuitMQP(BackendCircuitAQT):
    
    def set_token(self):
        try:
            import config
            self.token = config.lrz_key
        except:
            raise TequilaMQPException("No token found for MQP backend. Please set the token in a config.py.")
    
    def get_backend(self) -> MQPBackend:
        self.set_token()
        backend = None
        try:
            provider = MQPProvider(self.token)
            [backend] = provider.backends('AQT20')
            self.device = backend
        except Exception as e:
            raise TequilaMQPException(f"Invalid Token for MQP backend")
        return backend
  
   
    def get_circuit(self, circuit: Union[QuantumCircuit, list[QuantumCircuit]], qiskit_backend, initial_state=0, optimization_level=1,  *args, **kwargs) -> QuantumCircuit:
        circ = circuit.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
        circ = self.add_state_init(circ, initial_state)
        return circ
     
    
    def do_simulate(self, variables, initial_state=0, *args, **kwargs):
        raise TequilaMQPException("MQP backend does not support do_simulate")
    
    
class BackendExpectationValueMQP(BackendExpectationValueAQT):
    BackendCircuitType = BackendCircuitMQP

       
class TequilaMQPException(TequilaQiskitException):
    def __str__(self):
        return "Error in MQP backend:" + self.message
 