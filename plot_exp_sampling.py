import matplotlib.pyplot as plt
import pickle
import numpy as np

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
fig, ax = plt.subplots(len(sample_list) + 1, sharex=True, sharey=True)
for s, samples in enumerate(sample_list):
    with open("data/gdata2_{}.dat".format(samples), "rb") as file:
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
    
    #fig, ax = plt.subplots()

    ax[0].set_title("error bar")
    ax[0].errorbar(x, av, yerr=err, fmt='o', capsize=5, markersize=cs, color=color(samples), label='{} shots'.format(samples))
    
    ax[s+1].set_title("box plot for {} shots".format(samples))
    ax[s+1].boxplot(energy_diffs, patch_artist=True,  # fill with color
                   tick_labels=x)  # will be used to label x-ticks
    #plt[1].set_title("box plot")
    #plt[1].boxplot(energy_diffs, patch_artist=True,  # fill with color
    #               tick_labels=x)  # will be used to label x-ticks

    #plt.plot(x,av,label="av ({})".format(samples))
plt.legend()
plt.xlabel("number of Hydrogens")
plt.ylabel("finite sample energy")
plt.savefig("av_and_range_n.png")
plt.show()

exit(0)

#fig, ax = plt.subplots(len(energy_diff_results), sharex=True, sharey=True)
fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].set_title("error bar")
ax[0].boxplot(x, y, yerr=err, fmt='o', capsize=5, markersize=cs, color=color(samples), label='{} shots'.format(samples))

for i in range(len(energy_diff_results)):
    ax[i].boxplot(y, patch_artist=True,  # fill with color
                   tick_labels=x)  # will be used to label x-ticks
    ax[i].set_title("box plot for {} shots".format(sample_list[i]))
    ax[i].set_ylabel("energy diff")
plt.saveax("box_plot.png")
plt.show()


