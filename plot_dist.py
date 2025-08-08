import matplotlib.pyplot as plt
import numpy as np
import csv_to_data as csv2data
import sys
from plot_exp_sampling import upsampled
from calc_engs import calc_engs


points = np.linspace(0.5,2.5,50)


def color(x):
    if x==200:
        return "tab:orange"
    if x==400:
        return "tab:red"
    if x==800:
        return "tab:blue"
    if x==1000:
        return "navy"
    if x==2000:
        return "black"
    return None

energy_diff_results = []
x = []
samples_list = [200, 400, 800]
filename = "data/dist_data_200_mqp.csv"
try: 
    data = csv2data.csv_to_exp_data(filename)
except FileNotFoundError as e:
    print("ERROR: Couldn't find {}".format(filename))
    print(e)
    sys.exit(1)
# number of hydrogens
x = [a[0] for a in data]
x_unique = list(set(x))


dists = [a[1] for a in data]
# results of sampling
y = [a[-1] for a in data]

y = [[r for r in result_list if r != None] for result_list in y]

# exact energies
best = [a[3] for a in data]

for i in range(len(best)):
    if len(y[i]) == 0:
        y[i] = [best[i]] * 10 
energy_diffs = [list(map(lambda x: x - best[i], y[i])) for i in range(len(best))]
energy_diff_results.append(energy_diffs)

lo = [min(a) for a in y]
hi = [max(a) for a in y]
av = [sum(a)/len(a) for a in y] 


# 5 * 5 values for five molecules and five distances
lo = [lo[i] - best[i] for i in range(len(lo))]
hi = [hi[i] - best[i] for i in range(len(lo))]
av = [av[i] - best[i] for i in range(len(lo))]


# for every molecule
for i in range(len(x_unique)):
    # distances for current molecule 
    x_dist = [dists[j] for j in range(len(x)) if x[j] == x_unique[i]]
    
    lo_dist = [lo[j] for j in range(len(x)) if x[j] == x_unique[i]]
    hi_dist = [hi[j] for j in range(len(x)) if x[j] == x_unique[i]]
    av_dist = [av[j] for j in range(len(x)) if x[j] == x_unique[i]]
    
    best_dist = [best[j] for j in range(len(x)) if x[j] == x_unique[i]]
    
    plt.title("{} hydrogens".format(x_unique[i]))
    plt.plot(x_dist, lo_dist,'bo--', label="lo")
    plt.plot(x_dist,hi_dist,label="hi")
    plt.plot(x_dist,av_dist,label="av")
    plt.xlabel("H-H distance")
    plt.ylabel("energy difference")
    plt.legend()
    plt.show()

lo = [min(a) for a in y]
hi = [max(a) for a in y]
av = [sum(a)/len(a) for a in y] 

for i in range(len(x_unique)):
    x_dist = [dists[j] for j in range(len(x)) if x[j] == x_unique[i]]
    baseline = [calc_engs(x_unique[i], d) for d in points] 
    lo_dist = [lo[j] for j in range(len(x)) if x[j] == x_unique[i]]
    hi_dist = [hi[j] for j in range(len(x)) if x[j] == x_unique[i]]
    av_dist = [av[j] for j in range(len(x)) if x[j] == x_unique[i]]
    
    best_dist = [best[j] for j in range(len(x)) if x[j] == x_unique[i]]

    plt.title("{} hydrogens".format(x_unique[i]))
    plt.plot(x_dist, lo_dist, marker='o', label="lo")
    plt.plot(x_dist,hi_dist, marker='o', label="hi")
    plt.plot(x_dist,av_dist, marker='o', label="av")
    plt.plot(points,baseline,label="baseline", color="black", linestyle="--")
    plt.xlabel("H-H distance")
    plt.ylabel("energy")
    plt.legend()
    if x_unique[i] == 2:
        plt.savefig("h2_energies_range_2.svg", format="svg")

    plt.show()



for samples in samples_list:
    mults = samples // 200
    upsampled_gen = upsampled(y, mults)
    y_s = list(upsampled_gen)
    
    
    lo = [min(a) for a in y_s]
    hi = [max(a) for a in y_s]
    av = [sum(a)/len(a) for a in y_s] 

    for i in range(len(x_unique)):
        # distances for current molecule 
        x_dist = [dists[j] for j in range(len(x)) if x[j] == x_unique[i]]
        
        lo_dist = [lo[j] for j in range(len(x)) if x[j] == x_unique[i]]
        hi_dist = [hi[j] for j in range(len(x)) if x[j] == x_unique[i]]
        av_dist = [av[j] for j in range(len(x)) if x[j] == x_unique[i]]
        
        best_dist = [best[j] for j in range(len(x)) if x[j] == x_unique[i]]
        
        plt.title("{} hydrogens with {} samples".format(x_unique[i], samples))
        plt.plot(x_dist, lo_dist,label="lo")
        plt.plot(x_dist,hi_dist,label="hi")
        plt.plot(x_dist,av_dist,label="av")
        plt.xlabel("H-H distance")
        plt.ylabel("energy difference")
        plt.legend()
        if x_unique[i] == 2:
            plt.savefig("h2_energies_range_2.svg", format="svg")
        plt.show()

 
