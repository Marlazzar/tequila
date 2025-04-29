import matplotlib.pyplot as plt
import pickle
import numpy as np

# TODO: make 2 subfigures for all sample sizes
# one in which all the optimizer energies are shown and one which shows for every optimizer
# (1. achieved energy diff)
# 2. number of sampling calls
# 3. iterations needed/maxiters



def color(i):
    if x==1:
        return "tab:orange"
    if x==2:
        return "tab:red"
    if x==3:
        return "tab:blue"
    if x==4:
        return "navy"
    if x==5:
        return "black"
    return None

cs = 12
energy_diff_results = []
x = []
sample_list = [200]
maxiters = [10]
#fig, ax = plt.subplots(2*len(sample_list), sharex=True)
for s, samples in enumerate(sample_list):
        with open("data/optim_data_{}.dat".format(samples), "rb") as file:
            data = pickle.load(file)

        data = [a for a in data]
        # optimizer method 
        labels = [a[0] for a in data]
        # maxiter
        #m = [a[1] for a in data]

        # energies... a list of lists. the problem is that the lists can have varying lengths
        y = [a[-1] for a in data]
        x = list(range(len(y[0])))
        
        # exact energies
        best = [a[3] for a in data]

        # still a list of lists
        energy_diffs = [list(map(lambda x: x - best[i], y[i])) for i in range(len(y))]
        
        sample_calls = [a[2] for a in data]
        
        for i in range(len(y)):
            plt.plot(list(range(len(y[i]))), energy_diffs[i], label=labels[i], color=color(i))
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("energy diff")
        plt.title("energy diff for {} shots".format(samples))
        plt.show()
        
        
        fig, ax = plt.subplots(2)
        width = 0.25
        offset = width 
        x = np.arange(len(labels))
        
        sample_bars = ax[0].bar(x + offset, sample_calls, width, color="navy", label="sampling calls")
        ax[0].bar_label(sample_bars, padding=3)
        ax[0].set_title("sampling calls")
        ax[0].set_xticks(x+width, labels)
        ax[0].legend()
        ax[0].set_xlabel("optimizer method")
        ax[0].set_ylabel("sampling calls")
        
        iter_bars = ax[1].bar(x + offset, [len(e) for e in energy_diffs], width, color="tab:orange", label="iterations")
        ax[1].bar_label(iter_bars, padding=3)
        ax[1].set_title("iterations")
        ax[1].set_xticks(x+width, labels)
        ax[1].legend()
        ax[1].set_xlabel("optimizer method")
        ax[1].set_ylabel("iterations")

        plt.show()
#
#        ax[0].set_ylabel("energy diffs")
#        ax[0].set_title("error bars")
#        #fig.set_title("error bar")
#        ax[0].errorbar(x, av, yerr=err, fmt='o', capsize=5, markersize=cs, color=color(maxiter), label='{} maxiter'.format(maxiter))
#      
#        ax[m+1].set_title("#measurements")
#        ax[m+1].set_ylabel("sampling calls")
#        sample_calls_avgs = [sum(a)/len(a) for a in sample_calls]
#        ax[m+1].bar(x, sample_calls_avgs)  # will be used to label x-ticks

        #plt.plot(x,av,label="av ({})".format(samples))
exit(0)
plt.legend()
plt.xlabel("optimizer method")
plt.show()
fig, ax = plt.subplots(len(energy_diff_results), sharex=True, sharey=True)
for i in range(len(energy_diff_results)):
    ax[i].boxplot(energy_diff_results[i], patch_artist=True,  # fill with color
                   tick_labels=x)  # will be used to label x-ticks
    ax[i].set_title("box plot for {} shots".format(sample_list[i]))
    ax[i].set_ylabel("energy diff")
plt.savefig("box_plot.png")
plt.show()


