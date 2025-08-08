import sys
sys.path.append("src")
import src.tequila as tq
import numpy as np
import os
import csv
from src.tequila import QubitHamiltonian, QCircuit
from typing import Any
import pickle
import globals 
import subprocess
import csv

def coeff_sum(H):
    coeff_sum = 0
    for p in H.paulistrings:
        coeff_sum += abs(p.coeff)
    return coeff_sum


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
    
def make_hcb_grouping(H):
    H1 = tq.QubitHamiltonian()
    H2 = tq.QubitHamiltonian()
    H3 = tq.QubitHamiltonian()

    HOg1 = tq.QubitHamiltonian()
    HOg2 = tq.QubitHamiltonian()
    HOg3 = tq.QubitHamiltonian()

 
    U1 = tq.QCircuit()
    U2 = tq.gates.H([i for i in H.qubits])
    U3 = tq.gates.Rx(angle=-np.pi/2, target=[i for i in H.qubits])
    for p in H.paulistrings:
        q = p.naked().qubits
        if p.is_all_z():
          HOg1 += tq.QubitHamiltonian().from_paulistrings(p)
          H1 += tq.QubitHamiltonian().from_paulistrings(p)
        else:
          if (p.naked()[q[0]] == "X"):
              HOg2 += tq.QubitHamiltonian().from_paulistrings(p)
              for k, v in p.items():
                  p._data[k] = "Z"
              H2 += tq.QubitHamiltonian().from_paulistrings(p)
          else:
              HOg3 += tq.QubitHamiltonian().from_paulistrings(p)
              for k, v in p.items():
                  p._data[k] = "Z"
              H3 += tq.QubitHamiltonian().from_paulistrings(p)
    
    hamiltonians = [H1, H2, H3]
    circuits = [U1, U2, U3]
    suggested_samples = [None for _ in range(len(hamiltonians))]
    groups = [HOg1, HOg2, HOg3]
    result = [(H, U, G) for H, U, G in zip(hamiltonians, circuits, groups)]

    return result, suggested_samples

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


    filename  = "data/aqt_hcb_analysis.csv" 
    temp_uid_file = "data/aqt_analysis_temp.csv"
    E = tq.ExpectationValue(H=H_HCB, U=U_HCB, optimize_measurements=False)
    #v = {k:1.0 for k in E.extract_variables()}

    print("H_HCB", H_HCB)
    hcbs_groups, suggested_samples = make_hcb_grouping(H_HCB)


    result = tq.minimize(E, silent=True)
    exact_energy = result.energy
    v = result.variables
    print(v)
    exit(0)
   
    # TODO: save the results somehow
    # number of hs result_sample 
    results = []
    counts_list = []

    groups = ["Z", "X", "Y"]
    data = []
    i = 0
    while i < iterations:
        try:
            result = 0
            total_exact_energy = 0
            for j, group in enumerate(hcbs_groups):
                H = group[0]
                U = group[1]
                G = group[2]
                with open(temp_uid_file, "a") as file:
                    writer = csv.writer(file) 
                    writer.writerow([number_hs, groups[j]])

                Ehcb = tq.ExpectationValue(H=H, U=U_HCB + U)
                result_sampl_g = tq.simulate(Ehcb, variables=v, backend="mqp", samples=samples)
                
                exact_result = tq.minimize(Ehcb, silent=True)
                exact_energy = exact_result.energy

                print("result", result_sampl_g)
                print("counts", globals.counts)
                result += result_sampl_g
                total_exact_energy += exact_energy

                print("G", G)
                print("H", H)
                c_sum = coeff_sum(H)
                print("c_sum", c_sum)
                jobid = globals.jobid
                with open(filename, "a") as file:
                    writer = csv.writer(file) 
                    datarow = [number_hs, groups[j], G, jobid, globals.counts, c_sum, exact_energy, result_sampl_g]
                    data.append(datarow)
                    writer.writerow(datarow)
                print("result",result)
                print("exact energy", total_exact_energy)
            i = i + 1
        except Exception as e:
            print("Error in mqp sampling", e)
            # TODO: continue until we've collected #iterations results
            continue
        with open("data/tmp_data_{}_{}.txt".format(number_hs, samples), "a") as file:
            file.write(f"{number_hs}, {result}\n")
        results.append(result)
        
    print("exact energy", exact_energy)
    print("200 shots sampling energy", results)


    return exact_energy, results, data

if __name__ == "__main__":
    #tq.show_available_optimizers()
    subprocess.run(["mv data/tmp_data_* backup_datadata/"], shell=True)
    subprocess.run(["rm data/tmp_data_*"], shell=True)
    

    
    globals.init()
    dist_h = 1.0
    iterations = 10
    for samples in [200]:
        datalist = []
        filename = "data/exp_sampling_hcb_{}_jobs.csv".format(samples)
        if os.path.exists(filename):
            subprocess.run(["mv", filename, "backup_data/"], shell=True) 
        for i in range(4, 6, 2):
            try:
                exact_energy, results, data = linear_H(number_hs=i, dist_h=dist_h, samples=samples, iterations=iterations, static=True)
                datalist.append(data)
            except Exception as e:
                print("Error in mqp sampling", e)
                continue
        with open("data/exp_sampling_hcb.dat".format(samples), "wb") as file:
            pickle.dump(data,file) 
    print("numbher h, dist_h, exact_energy, results")
 