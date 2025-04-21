from calendar import c
from tequila.circuit.circuit import QCircuit
from tequila.utils.keymap import KeyMapRegisterToSubregister
from tequila import BitString, BitNumbering, BitStringLSB
from tequila.circuit.compiler import change_basis
import numbers, typing, numpy, copy, warnings
from tequila.objective.objective import Variable, format_variable_dictionary
import typing
from tequila.utils.misc import to_float
import numpy
from typing import Union
from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit, TequilaQiskitException
from mqp.qiskit_provider import MQPProvider, MQPBackend
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import AQTResource
from qiskit_aqt_provider.primitives import AQTEstimator, AQTSampler
import qiskit
from qiskit.circuit import QuantumCircuit
import qiskit_aqt_provider
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException, TequilaWarning, circuit





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
    
    def sample(self, variables, samples, read_out_qubits=None, circuit=None, initial_state=0, *args, **kwargs):
        if initial_state != 0 and not self.supports_sampling_initialization:
            raise TequilaException("Backend does not support initial states for sampling")

        if isinstance(initial_state, QubitWaveFunction) and not self.supports_generic_initialization:
            raise TequilaException("Backend does not support arbitrary initial states")

        self.update_variables(variables)
        if read_out_qubits is None:
            read_out_qubits = self.abstract_qubits

        if len(read_out_qubits) == 0:
            raise Exception("read_out_qubits are empty")

        if circuit is None:
            circuit = self.add_measurement(circuit=self.circuit, target_qubits=read_out_qubits)
        else:
            if isinstance(circuit, list):
                assert len(circuit) == len(read_out_qubits), "circuit and read_out_qubits have to be of the same length"
                for i, c in enumerate(circuit):
                    circuit[i] = self.add_measurement(circuit=c, target_qubits=read_out_qubits[i])
            else:
                circuit = self.add_measurement(circuit=circuit, target_qubits=read_out_qubits)
        return self.do_sample(samples=samples, circuit=circuit, read_out_qubits=read_out_qubits,
                              initial_state=initial_state, *args, **kwargs)

    
    
    def do_sample(self, circuit: Union[QuantumCircuit, list[QuantumCircuit]], samples: int, read_out_qubits, initial_state=0, *args,
                  **kwargs) -> Union[QubitWaveFunction, list[QubitWaveFunction]]:
        optimization_level = 1
        if 'optimization_level' in kwargs:
            optimization_level = kwargs['optimization_level']
        qiskit_backend = self.retrieve_device(self.device)
        if isinstance(circuit, list):
            for i, c in enumerate(circuit):
                circuit[i] = c.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
                circuit[i] = self.add_state_init(c, initial_state)
                basis = qiskit_backend.operation_names
                circuit[i] = qiskit.transpile(c, backend=qiskit_backend, basis_gates=basis, optimization_level=optimization_level)
            # batch jobs
            job = qiskit_backend.run(circuit, shots=samples)
            counts = job.result().get_counts()
            wfns = []
            for i, count in enumerate(counts):
                wfn = self.convert_measurements(count, target_qubits=read_out_qubits[i])
                wfns.append(wfn)
            return wfns
        
        circuit = circuit.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
        circuit = self.add_state_init(circuit, initial_state)   
        circuit = qiskit.transpile(circuit, backend=qiskit_backend, optimization_level=optimization_level)
        job = qiskit_backend.run(circuit, shots=samples)
        return self.convert_measurements(job.result().get_counts(), target_qubits=read_out_qubits)

    def convert_measurements(self, qiskit_counts, target_qubits=None) -> list[QubitWaveFunction]:
        result = QubitWaveFunction(self.n_qubits, self.numbering)
        # todo there are faster ways
        for k, v in qiskit_counts.items():
            # Qiskit uses LSB bitstrings, but from_binary expects MSB
            converted_key = BitString.from_binary(k[::-1])
            result[converted_key] = v
        if target_qubits is not None:
            mapped_target = [self.qubit_map[q].number for q in target_qubits]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(subregister=mapped_target, register=mapped_full)
            result = QubitWaveFunction.from_wavefunction(result, keymap, n_qubits=len(target_qubits))
   
        return result

    # TODO: do i need multiple variables here?
    # expects a list of paulistrings
    def sample_batches(self, samples: int, groups, variables, initial_state: Union[int, QubitWaveFunction] = 0,
                    
                           *args, **kwargs) -> numbers.Real:
       # the circuits for the batch
        circuits = []
        # readout qubits for the batch
        read_out_qubits_list = []
        for group in groups:
            # TODO: self.abstract_qubits for every group the same? yes if theyre all from the same hamiltonian 
            not_in_u = [q for q in group.qubits if q not in self.abstract_qubits]
            reduced_ps = group.trace_out_qubits(qubits=not_in_u)
            if reduced_ps.coeff == 0.0:
                return 0.0
            if len(reduced_ps._data.keys()) == 0:
                return reduced_ps.coeff
    
            # make basis change and translate to backend
            basis_change = QCircuit()
            qubits = []
            for idx, p in reduced_ps.items():
                qubits.append(idx)
                basis_change += change_basis(target=idx, axis=p)
    
            # add basis change to the circuit
            # deepcopy is necessary to avoid changing the circuits
            # can be circumvented by optimizing the measurements
            # on construction: tq.ExpectationValue(H=H, U=U, optimize_measurements=True)
            circuit = self.create_circuit(circuit=copy.deepcopy(self.circuit), abstract_circuit=basis_change)
            circuits.append(circuit)
            read_out_qubits_list.append(qubits)
            # run simulators
        # somehow call the sample method for batches
        wfns = self.sample(samples=samples, circuit=circuits, read_out_qubits=read_out_qubits_list, variables=variables,
                             initial_state=initial_state, *args, **kwargs)
        
        # TODO: count is now a list of dictionaries for every ps/group
        # compute energy

        # TODO: this is creating a single result, so it will only work for batching the paulistrings of 
        # a single hamiltonian
        E = 0.0
        n_samples = 0
        for i, wfn in enumerate(wfns): 
            E_tmp = 0.0
            for key, count in wfn.items():
                parity = key.array.count(1)
                sign = (-1) ** parity
                E_tmp += sign * count
                n_samples += count
                E_tmp = E_tmp / samples * groups[i].coeff
                 
            E += E_tmp
        assert n_samples == samples * len(wfns)
        return E



    
class BackendExpectationValueAQT(BackendExpectationValueQiskit):
    BackendCircuitType = BackendCircuitAQT


    def extract_variables(self) -> typing.Dict[str, numbers.Real]:
        """
        wrapper over circuit extract variables
        Returns
        -------
        Dict
            Todo: is it really a dict?
        """
        result = []
        if self.U is not None:
            result = self.U.extract_variables()
        return result

    def __init__(self, E, variables, noise, device, *args, **kwargs):
        """

        Parameters
        ----------
        E:
            the uncompiled expectationvalue
        variables:
            variables for compilation of circuit
        noise:
            noisemodel for compilation of circuit
        device:
            device for compilation of circuit
        """
        self.abstract_expectationvalue = E
        self._input_args = {"variables": variables, "device": device, "noise": noise, **kwargs}
        self._U = self.initialize_unitary(E.U, variables=variables, noise=noise, device=device, **kwargs)
        self._reduced_hamiltonians = self.reduce_hamiltonians(self.abstract_expectationvalue.H)
        self._H = self.initialize_hamiltonian(self._reduced_hamiltonians)
        self._variables = E.extract_variables()
        self._contraction = E._contraction
        self._shape = E._shape



    def __call__(self, variables, samples: int = None, initial_state: Union[int, QubitWaveFunction] = 0, batching: bool = False, *args, **kwargs):

        variables = format_variable_dictionary(variables=variables)
        if self._variables is not None and len(self._variables) > 0:
            if variables is None or (not set(self._variables) <= set(variables.keys())):
                raise TequilaException(
                    "BackendExpectationValue received not all variables. Circuit depends on variables {}, you gave {}".format(
                        self._variables, variables))

        if samples is None:
            data = self.simulate(variables=variables, initial_state=initial_state, *args, **kwargs)
        else:
            data = self.sample(variables=variables, samples=samples, initial_state=initial_state, batching=batching, *args, **kwargs)

        if self._shape is None and self._contraction is None:
            # this is the default
            return numpy.sum(data)

        if self._shape is not None:
            data = data.reshape(self._shape)
        else:
            data = float(data)
        if self._contraction is None:
            return data
        else:
            return self._contraction(data)

    def sample(self, variables, samples, initial_state: Union[int, QubitWaveFunction] = 0, batching: bool = False, *args, **kwargs) -> numpy.array:
        """
        sample the expectationvalue.

        Parameters
        ----------
        variables: dict:
            variables to supply to the unitary.
        samples: int:
            number of samples to perform.
        initial_state: int or QubitWaveFunction:
            the initial state of the circuit
        args
        kwargs

        Returns
        -------
        numpy.ndarray:
            a numpy array, the result of sampling.
        """

        suggested = None
        if hasattr(samples, "lower") and samples.lower()[:4] == "auto":
            if self.abstract_expectationvalue.samples is None:
                raise TequilaException("samples='auto' requested but no samples where set in individual expectation values")
            total_samples = int(samples[5:])
            samples = max(1, int(self.abstract_expectationvalue.samples * total_samples))
            suggested = samples
            # samples are not necessarily set (either the user has to set it or some functions like optimize_measurements)

        if suggested is not None and suggested != samples:
            warnings.warn("simulating with samples={}, but expectationvalue carries suggested samples={}\nTry calling with samples='auto-total#ofsamples'".format(samples, suggested), TequilaWarning)

        self.update_variables(variables)

        result = []
        for H in self._reduced_hamiltonians:
            E = 0.0
            if len(H.qubits) == 0:
                E = sum([ps.coeff for ps in H.paulistrings])
            elif H.is_all_z():
                E = self.U.sample_all_z_hamiltonian(samples=samples, hamiltonian=H, variables=variables, initial_state=initial_state,
                                                *args, **kwargs)
            else:
                groups = []
                for ps in H.paulistrings:
                    groups.append(ps)
                if batching:
                    E = self.U.sample_batches(samples=samples, groups=groups, variables=variables, initial_state=initial_state)
                else: 
                    for g in groups:
                        E += self.U.sample_paulistring(samples=samples, paulistring=g, variables=variables, initial_state=initial_state,
                                                   *args, **kwargs)
            result.append(to_float(E))
        return numpy.asarray(result)

    