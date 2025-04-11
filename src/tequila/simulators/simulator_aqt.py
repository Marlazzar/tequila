from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit, TequilaQiskitException
from mqp.qiskit_provider import MQPProvider, MQPBackend
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import AQTResource
from qiskit_aqt_provider.primitives import AQTEstimator, AQTSampler
import qiskit
from qiskit.circuit import QuantumCircuit
import qiskit_aqt_provider
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException, TequilaWarning





class TequilaAQTException(TequilaQiskitException):
    def __str__(self):
        return "Error in AQT backend:" + self.message


class BackendCircuitAQT(BackendCircuitQiskit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.get_backend()
    
        
    def get_backend(token: str = "") -> AQTResource:
        provider = AQTProvider(token)
        # TODO: what if somebody actually passes a valid aqt cloud token?
        backend = provider.get_backend('offline_simulator_no_noise')
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
        qiskit_backend = self.retrieve_device(self.device)
        circuit = circuit.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
        circuit = self.add_state_init(circuit, initial_state)
        circuit = qiskit.transpile(circuit, backend=qiskit_backend,
                                           optimization_level=optimization_level
                                           )
        job = qiskit_backend.run(circuit, shots=samples)
        return self.convert_measurements(job, target_qubits=read_out_qubits)
    

class BackendExpectationValueAQT(BackendExpectationValueQiskit):
    BackendCircuitType = BackendCircuitAQT