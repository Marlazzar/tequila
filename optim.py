import sys
sys.path.append("src")
import src.tequila as tq
import csv
import numpy as np
from src.tequila import QubitHamiltonian, QCircuit
from typing import Any
import pylab
import subprocess
import os
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
    
    # TODO: iterations  , energy convergence
    results = []
    sample_calls = []
    energies = None
    print(optimizer)
    for i in range(iterations):
            globals.calls = 0
            print(globals.calls)
            result_sampl = tq.minimize(E, **optimizer, backend="aqt", samples=samples, hcb=True)
        #except Exception as e:
        #    print("error in mqp backend")
        #    exit(1)
        #with open("data/optim_tmp_data_{}_{}.txt".format(number_hs, samples), "a") as file:
        #    file.write(f"{number_hs}, {result_sampl}\n")
            best_energy = result_sampl.energy
            if i == 0:
                print(maxiter)
                method = optimizer["method"]
                gradient=""
                if "gradient" in optimizer.keys():
                    gradient = optimizer["gradient"]
                result_sampl.history.plot("energies", filename=f"figures/{method}{gradient}_{samples}")
                #result_sampl.history.plot("energies", labels=optimizer["method"])
            energies = result_sampl.history.energies
     
    print("exact energy", exact_energy)
    print("200 shots sampling energy", results)
    print("sampling calls", sample_calls)

    return exact_energy, results, globals.calls, energies

def get_optimizer(method, maxiter):
    # need gradients
    if method == "bfgs":
        return {
            "method": "bfgs",
            "maxiter": maxiter,
            "gradient": "2-point",
            #"lr": 0.01
        }
    elif method == "sgd_qng":
        return {
            "method": "sgd",
            "maxiter": maxiter,
            "gradient": "qng",
            "lr": 0.01
        }
    elif method == "sgd":
        return {
            "method": "sgd",
            "maxiter": maxiter,
            "lr": 0.01
        }
    elif method == "adam_qng":
        return {
            "method": "adam",
            "maxiter": maxiter,
            "gradient": "qng",

        }
    elif method == "adam":
        return {
            "method": "adam",
            "maxiter": maxiter,
            "gradient": "2-point",

        }

    # needs hessian and gradient - doesn't work with numerical 2-point, needs jacobian
    elif method == "newton-cg":
        return {
            "method": "newton-cg",
            "maxiter": maxiter,
        }
    # these all don't need gradients
    elif method == "cobyla":
        return {
            "method": "cobyla",
            "maxiter": maxiter
        }
    elif method == "slsqp":
        return {
            "method": "slsqp",
            "maxiter": maxiter
        }
    elif method == "nelder-mead":
        return {
            "method": "nelder-mead",
            "maxiter": maxiter
        }
    
    else:
        raise ValueError("Unknown optimizer method")


if __name__ == "__main__":
    samples_list = [200]
    data = []
    maxiters = [10]
       #for s in samples:
    #optimizers = [ "cobyla", "nelder-mead", "adam", "sgd", "sgd_qng", "newton-cg"]
    optimizers =["cobyla" ]
    for samples in samples_list:
        for maxiter in maxiters:
            filename = "data/optim_data_{}_{}.csv".format(samples, maxiter)
            if os.path.exists(filename):
               subprocess.run(["mv", filename, "data_backup/"]) 
            for o in optimizers:
                optimizer = get_optimizer(o, maxiter=maxiter)
                try: 
                    exact_energy, results, sample_calls, energies = linear_H(number_hs=2, dist_h=1.0, samples=samples, iterations=1, static=True, optimizer=optimizer)
                    data.append((o, maxiter, sample_calls, exact_energy, energies))
                    print("error in optimizer", o, "with maxiter", maxiter)
                    with open(filename, "a") as file:
                        writer = csv.writer(file)
                        writer.writerow([o, maxiter, sample_calls, exact_energy, *energies])
                except Exception as e:
                    print(e)
                    continue
            filename = "data/optim_data_{}_{}.dat".format(samples, maxiter)
            if os.path.exists(filename):
                subprocess.run(["mv", filename, "data_backup/"])
            with open("data/optim_data_{}.dat".format(samples), "wb") as file:
                pickle.dump(data,file) 
            print("optimizer, maxiter, sample_calls, exact_energy, energies")
            print("data", data)