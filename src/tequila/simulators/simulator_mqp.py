from mqp.qiskit_provider import MQPProvider, MQPBackend
from qiskit.circuit import QuantumCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit, TequilaQiskitException
import qiskit


class BackendCircuitMQP(BackendCircuitQiskit):
    
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
    
    def do_sample(self, circuit: QuantumCircuit, samples: int, read_out_qubits, initial_state=0, *args,
                  **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing sampling.
        Parameters
        ----------
        circuit: qiskit.QuantumCircuit:
            the circuit from which to sample.
        samples:
            the number of samples to take.
        initial_state:
            initial state of the circuit
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of sampling.
        """
        optimization_level = 1
        if 'optimization_level' in kwargs:
            optimization_level = kwargs['optimization_level']
        qiskit_backend = self.get_backend()
        circuit = circuit.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
        circuit = self.add_state_init(circuit, initial_state)
        basis = qiskit_backend.operation_names
        # coupling_map?
        circuit = qiskit.transpile(circuit, backend=qiskit_backend, basis_gates=basis, coupling_map=coupling_map,
                                           optimization_level=optimization_level
                                           )
        job = qiskit_backend.run(circuit, shots=samples, queued=True)
        # TODO: the program will get stuck forever in the convert_measurements method, because the mqp qpu doesn't execute it 
        # immediately. We should probably print out the job uuid, print some instructions for how to get the result and then 
        # just exit
        return self.convert_measurements(job, target_qubits=read_out_qubits)
 
    def do_simulate(self, variables, initial_state=0, *args, **kwargs):
        raise TequilaMQPException("MQP backend does not support do_simulate")
    
    
class BackendExpectationValueMQP(BackendExpectationValueQiskit):
    BackendCircuitType = BackendCircuitMQP
    
class TequilaMQPException(TequilaQiskitException):
    def __str__(self):
        return "Error in MQP backend:" + self.message
 