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

# TODO: mit aqt mergen

def make_hcb_grouping(H):
    H1 = QubitHamiltonian()
    H2 = QubitHamiltonian()
    H3 = QubitHamiltonian()
    
    U1 = QCircuit()
    U2 = gates.H([i for i in H.qubits])
    U3 = gates.Rx(angle=-np.pi/2, target=[i for i in H.qubits])
    for p in H.paulistrings:
      q = p.naked().qubits
      if p.is_all_z():
          H1 += QubitHamiltonian().from_paulistrings(p)
      else:
          if (p.naked()[q[0]] == "X"):
              for k, v in p.items():
                  p._data[k] = "Z"
              H2 += QubitHamiltonian().from_paulistrings(p)
          else:
              for k, v in p.items():
                  p._data[k] = "Z"
              H3 += QubitHamiltonian().from_paulistrings(p)
    
    hamiltonians = [H1, H2, H3]
    circuits = [U1, U2, U3]
    return hamiltonians, circuits



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
        qiskit_backend = self.get_backend()
        sampling_circuits = []
        k = 1
        shots = samples
        if samples > 200:
            k = int(samples / 200)
            shots = 200
        w = QubitWaveFunction(self.n_qubits, self.numbering)
        if isinstance(circuit, list):
            for i, c in enumerate(circuit):
                circ = c.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
                circ = self.add_state_init(circ, initial_state)
                #circ = qiskit.transpile(circ, backend=qiskit_backend, basis_gates=basis, optimization_level=optimization_level)
                sampling_circuits.extend([circ] * k)
            if len(sampling_circuits) > 100:
                raise TequilaMQPException("Too many circuits to sample. Please reduce the number of circuits or increase the number of shots.")
            # batch jobs
            job = qiskit_backend.run(sampling_circuits, shots=shots, queued=True)
            counts = job.result().get_counts()
            wfns = []
            for i, count in enumerate(counts):
                wfn = self.convert_measurements(count, target_qubits=read_out_qubits[i // k])
                wfns.append(wfn)
            wfns = [sum(wfns[i:i+k], w) for i in range(0, len(wfns), k)]   

            return wfns
        
        circuit = circuit.assign_parameters(self.resolver)  # this is necessary -- see qiskit-aer issue 1346
        circuit = self.add_state_init(circuit, initial_state)   
        #circuit = qiskit.transpile(circuit, backend=qiskit_backend, optimization_level=optimization_level)
        sampling_circuits = [circuit] * k
        job = qiskit_backend.run(sampling_circuits, shots=shots, queued=True)
        counts = job.result().get_counts()
        if isinstance(counts, list):
            wfns = []
            for i, count in enumerate(counts):
                wfn = self.convert_measurements(count, target_qubits=read_out_qubits)
                wfns.append(wfn)
            wfn = sum(wfns, w)
        else:
            wfn = self.convert_measurements(counts, target_qubits=read_out_qubits)
        return wfn
    
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

    # TODO: maybe check if a pauli string is all z and skip the basis change if so
    def sample_batches(self, samples: int, groups, variables, initial_state: Union[int, QubitWaveFunction] = 0, circuits: list[QuantumCircuit] = None,
                           *args, **kwargs) -> numbers.Real:
        # either dont pass circuits to just create some for the basis change of paulistrings
        # or pass an entire circuit for each group
       # the circuits for the batch
        if circuits is None:
            circuits = []
        else:
            assert len(circuits) == len(groups), "circuits and groups have to be of the same length"
        # readout qubits for the batch
        read_out_qubits_list = []
        E = 0.0
        sampling_groups = []
        for group in groups:
            # TODO: self.abstract_qubits for every group the same? yes if theyre all from the same hamiltonian 
            not_in_u = [q for q in group.qubits if q not in self.abstract_qubits]
            reduced_ps = group.trace_out_qubits(qubits=not_in_u)
            if reduced_ps.coeff == 0.0:
                continue

            if len(reduced_ps._data.keys()) == 0:
                E += reduced_ps.coeff
                continue
    
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
            sampling_groups.append(group)
            # run simulators
        # somehow call the sample method for batches
        wfns = self.sample(samples=samples, circuit=circuits, read_out_qubits=read_out_qubits_list, variables=variables,
                             initial_state=initial_state, *args, **kwargs)
        

        # TODO: this is creating a single result, so it will only work for batching the paulistrings of 
        # a single hamiltonian
        n_samples = 0
        for i, wfn in enumerate(wfns): 
            E_tmp = 0.0
            for key, count in wfn.items():
                parity = key.array.count(1)
                sign = (-1) ** parity
                E_tmp += sign * count
                n_samples += count
            E_tmp = E_tmp / samples * sampling_groups[i].coeff
            E += E_tmp
        assert n_samples == samples * len(wfns)
        return E
    
    # TODO: hcb will always be batched right now
    def sample_hcb_batch(self, samples: int, hamiltonians, basis_changes, variables, initial_state: Union[int, QubitWaveFunction] = 0, *args, **kwargs) -> numbers.Real:
        assert len(basis_changes) == len(hamiltonians) , "circuits and hamiltonians both need to have length 3"
        # readout qubits for the batch
        read_out_qubits_list = []
        circuits = []
        hamiltonians_tmp = []
        E = 0.0
        for i, hamiltonian in enumerate(hamiltonians):
            abstract_qubits_H = hamiltonian.qubits
            if len(abstract_qubits_H) == 0:
                E += sum([ps.coeff for ps in hamiltonian.paulistrings])
                continue
            hamiltonians_tmp.append(hamiltonian)
            # assert that the Hamiltonian was mapped before
            if not all(q in self.qubit_map.keys() for q in abstract_qubits_H):
                raise TequilaException(
                "Qubits in {}-qubit Hamiltonian were not traced out for {}-qubit circuit".format(hamiltonian.n_qubits,
                                                                                                 self.n_qubits))

            circuit = self.create_circuit(circuit=copy.deepcopy(self.circuit), abstract_circuit=basis_changes[i])
            circuits.append(circuit)
            read_out_qubits_list.append(abstract_qubits_H)
            # run simulators
        # somehow call the sample method for batches
        wfns = self.sample(samples=samples,circuit=circuits, read_out_qubits=read_out_qubits_list, variables=variables,
                             initial_state=initial_state, *args, **kwargs)
        
        for i, hamiltonian in enumerate(hamiltonians_tmp):  
            abstract_qubits_H = hamiltonian.qubits        
            read_out_map = {q: i for i, q in enumerate(abstract_qubits_H)}

            for paulistring in hamiltonian.paulistrings:
                Etmp = 0.0
                n_samples = 0
                for key, count in wfns[i].items():
                    # get all the non-trivial qubits of the current PauliString (meaning all Z operators)
                    # and mapp them to the backend qubits
                    mapped_ps_support = [read_out_map[i] for i in paulistring._data.keys()]
                    # count all measurements that resulted in |1> for those qubits
                    parity = [k for i, k in enumerate(key.array) if i in mapped_ps_support].count(1)
                    # evaluate the PauliString
                    sign = (-1) ** parity
                    Etmp += sign * count
                    n_samples += count
                E += (Etmp / samples) * paulistring.coeff
                # small failsafe
                assert n_samples == samples 
        return E



    def do_simulate(self, variables, initial_state=0, *args, **kwargs):
        raise TequilaMQPException("MQP backend does not support do_simulate")
    
    
class BackendExpectationValueMQP(BackendExpectationValueQiskit):
    BackendCircuitType = BackendCircuitMQP

    def sample(self, variables, samples, initial_state: Union[int, QubitWaveFunction] = 0, hcb: bool = False, batching: bool = False, *args, **kwargs) -> numpy.array:
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
                if hcb:
                    hamiltonians, basis_changes = make_hcb_grouping(H)
                    E = self.U.sample_hcb_batch(samples=samples, hamiltonians=hamiltonians, basis_changes=basis_changes, variables=variables, initial_state=initial_state)
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

    
class TequilaMQPException(TequilaQiskitException):
    def __str__(self):
        return "Error in MQP backend:" + self.message
 