import sys
sys.path.append("src")
import src.tequila as tq
import numpy as np
from src.tequila import QubitHamiltonian, QCircuit
from typing import Any
import pylab
import pickle

def make_guess(n, edges):
    guess = np.eye(n)
    for i, j in edges:
        guess[i][j] = 1.0
        guess[j][i] = -1.0
    return guess

def make_geometry(number_hs, dist_h):
    geometry = ""
    for i in range(number_hs):
        geometry += f"h 0.0 0.0 {i*dist_h}\n"
    return geometry
    
    
def linear_H(number_hs, dist_h=1.0, samples=200, iterations=1, static:bool = False)  \
        -> tuple[QubitHamiltonian, QCircuit, float, float, Any]:
    geometry = make_geometry(number_hs=number_hs, dist_h=dist_h)
    
    print(geometry)
    mol = tq.Molecule(geometry=geometry, basis_set="sto-3g", transformation="ReorderedJordanWigner")
        
    mol = mol.use_native_orbitals()
    edges = [(2*i, 2*i +1) for i in range(number_hs // 2)]
    # guess initial 
    guess = make_guess(number_hs, edges=edges)
    U_HCB  = mol.make_ansatz(name="HCB-SPA", edges=edges)
    opt = tq.chemistry.optimize_orbitals(mol, circuit=U_HCB, initial_guess=guess.T,silent=True, use_hcb=True)
    H_HCB = opt.molecule.make_hardcore_boson_hamiltonian() # name="HCB-SPA"

    E = tq.ExpectationValue(H=H_HCB, U=U_HCB)
    #v = {k:1.0 for k in E.extract_variables()}
    
    result = tq.minimize(E, silent=True)
    exact_energy = result.energy
    v = result.variables
   
    # TODO: save the results somehow
    # number of hs result_sample 
    results = []
    for i in range(iterations):
        try:
            result_sampl = tq.simulate(E, variables=v, backend="aqt", samples=samples, hcb=True)
        except Exception as e:
            print("error in mqp backend")
            exit(1)
        with open("data/tmp_data_{}_{}.txt".format(number_hs, samples), "a") as file:
            file.write(f"{number_hs}, {result_sampl}\n")
        results.append(result_sampl)
        
    print("exact energy", exact_energy)
    print("200 shots sampling energy", results)


    return exact_energy, results


if __name__ == "__main__":
    #tq.show_available_optimizers()
    samples = 200
    dist_h = 1.0
    for samples in [200]:
        data = []
        for i in range(2, 4, 2):
            exact_energy, results = linear_H(number_hs=i, dist_h=dist_h, samples=samples, iterations=10, static=True)
            data.append((i, dist_h, samples, exact_energy, results))
        with open("data/gdata2_{}.dat".format(samples), "wb") as file:
            pickle.dump(data,file) 
    print("numbher h, dist_h, samples, exact_energy, results")
    print("data", data)