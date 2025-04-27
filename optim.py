import sys
sys.path.append("src")
import src.tequila as tq
import numpy as np
from src.tequila import QubitHamiltonian, QCircuit
from typing import Any
import pylab
import pickle
import globals 

from navid_main import make_geometry, make_guess

    
def linear_H(number_hs, dist_h=1.0, samples=200, iterations=1, static:bool = False, optimizer: dict = { "method": "bfgs", "maxiter": 100 })  \
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
   
    results = []
    sample_calls = []
    print(optimizer)
    for i in range(iterations):
        #try:
            globals.calls = 0
            print(globals.calls)
            result_sampl = tq.minimize(E, **optimizer, backend="aqt", samples=samples, hcb=True)
        #except Exception as e:
        #    print("error in mqp backend")
        #    exit(1)
        #with open("data/optim_tmp_data_{}_{}.txt".format(number_hs, samples), "a") as file:
        #    file.write(f"{number_hs}, {result_sampl}\n")
            best_energy = result_sampl.energy
            calls = globals.calls
            print(calls)
            #best_angles = result_sampl.angles
            results.append(best_energy)
            sample_calls.append(calls)
        
    print("exact energy", exact_energy)
    print("200 shots sampling energy", results)
    print("sampling calls", sample_calls)

    return exact_energy, results, sample_calls

def get_optimizer(method, maxiter):
    if method == "bfgs":
        return {
            "method": "bfgs",
            "maxiter": 100,
            #"gradient": "qng",
            #"lr": 0.01
        }
    elif method == "adam":
        return {
            "method": "adam",
            "maxiter": 100
        }
    elif method == "cobyla":
        return {
            "method": "cobyla",
            "maxiter": 100
        }
    else:
        raise ValueError("Unknown optimizer method")
    return {
        "method": method,
        "maxiter": 100
    }
    
if __name__ == "__main__":
    tq.show_available_optimizers()
    exit(0)
    samples_list = [200, 400, 800]
    data = []
    maxiters = [10, 50, 100]
       #for s in samples:
    optimizers = [ "bfgs", "adam", "cobyla"]
    for samples in samples_list:
        for maxiter in maxiters:
            for o in optimizers:
                optimizer = get_optimizer(o, maxiter=maxiter)
                exact_energy, results, sample_calls = linear_H(number_hs=2, dist_h=1.0, samples=samples, iterations=10, static=True, optimizer=optimizer)
                data.append((o, maxiter, sample_calls, exact_energy, results))
            with open("data/optim_data_{}.dat".format(samples), "wb") as file:
                pickle.dump(data,file) 
            print("optimizer, maxiter, sample_calls, exact_energy, results")
            print("data", data)