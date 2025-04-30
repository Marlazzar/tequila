import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv_to_data as csv2data

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

cs = 12
energy_diff_results = []
x = []
sample_list = [200, 400, 800]
fig, ax = plt.subplots(len(sample_list), sharex=True)
for s, samples in enumerate(sample_list):
    #with open("data/exp_sampling_{}.dat".format(samples), "rb") as file:
    #    data = pickle.load(file)
    data = csv2data.csv_to_exp_data("data/exp_sampling_{}.csv".format(samples))
    # number of hydrogens
    x = [a[0] for a in data]
    # results of sampling
    y = [a[-1] for a in data]
    
    y = [[r for r in result_list if r != None] for result_list in y]
    
     
    # exact energies
    best = [a[-2] for a in data]
    
    for i in range(len(best)):
        if len(y[i]) == 0:
            y[i] = [best[i]] * 10 
    energy_diffs = [list(map(lambda x: x - best[i], y[i])) for i in range(len(best))]
    energy_diff_results.append(energy_diffs)
    
    lo = [min(a) for a in y]
    hi = [max(a) for a in y]
    av = [sum(a)/len(a) for a in y]
    
    lo = [lo[i] - best[i] for i in range(len(lo))]
    hi = [hi[i] - best[i] for i in range(len(lo))]
    av = [av[i] - best[i] for i in range(len(lo))]
    err = [max(abs(lo[i]),abs(hi[i])) for i in range(len(lo))]
    #plt.plot(x,lo,label="lo ({})".format(samples))
    #plt.plot(x,hi,label="hi ({})".format(samples))
    cs = cs -2
    
    #fig, ax = plt.subplots()

    ax[s].set_title("box plot for {} shots".format(samples))
    ax[s].boxplot(energy_diffs, patch_artist=True,  # fill with color
                   tick_labels=x)  # will be used to label x-ticks
    ax[s].legend()
    ax[s].set_xlabel("number of Hydrogens")
    ax[s].set_ylabel("finite sample energy")

    #plt[1].set_title("box plot")
    #plt[1].boxplot(energy_diffs, patch_artist=True,  # fill with color
    #               tick_labels=x)  # will be used to label x-ticks

    #plt.plot(x,av,label="av ({})".format(samples))
plt.show()
for s, samples in enumerate(sample_list):
    with open("data/exp_sampling_{}.dat".format(samples), "rb") as file:
        data = pickle.load(file)
    
    # number of hydrogens
    x = [a[0] for a in data]
    # results of sampling
    y = [a[-1] for a in data]
    
    y = [[r for r in result_list if r != None] for result_list in y]
    
     
    # exact energies
    best = [a[-2] for a in data]
    
    for i in range(len(best)):
        if len(y[i]) == 0:
            y[i] = [best[i]] * 10 
    energy_diffs = [list(map(lambda x: x - best[i], y[i])) for i in range(len(best))]
    energy_diff_results.append(energy_diffs)
    
    lo = [min(a) for a in y]
    hi = [max(a) for a in y]
    av = [sum(a)/len(a) for a in y]
    
    lo = [lo[i] - best[i] for i in range(len(lo))]
    hi = [hi[i] - best[i] for i in range(len(lo))]
    av = [av[i] - best[i] for i in range(len(lo))]
    err = [max(abs(lo[i]),abs(hi[i])) for i in range(len(lo))]
    #plt.plot(x,lo,label="lo ({})".format(samples))
    #plt.plot(x,hi,label="hi ({})".format(samples))
    cs = cs -2
 
    plt.title("error bar")
    plt.errorbar(x, av, yerr=err, fmt='o', capsize=5, markersize=cs, color=color(samples), label='{} shots'.format(samples))

plt.legend()
plt.xlabel("number of Hydrogens")
plt.ylabel("finite sample energy")
plt.show()
    
