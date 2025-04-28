import matplotlib.pyplot as plt
import pickle

def color(x):
    if x==10:
        return "tab:orange"
    if x==50:
        return "tab:red"
    if x==100:
        return "tab:blue"
    if x==1000:
        return "navy"
    if x==2000:
        return "black"
    return None

cs = 12
energy_diff_results = []
x = []
sample_list = [200]
maxiters = [10]
fig, ax = plt.subplots(2, sharex=True)
for s, samples in enumerate(sample_list):
    for m, maxiter in enumerate(maxiters):
        with open("data/optim_data_{}.dat".format(samples), "rb") as file:
            data = pickle.load(file)

        data = [a for a in data if a[1] == maxiter]
        # optimizer method 
        x = [a[0] for a in data]
        # maxiter
        #m = [a[1] for a in data]

        # results of sampling
        y = [a[-1] for a in data]
        print(len(y))
        # exact energies
        best = [a[-2] for a in data]

        energy_diffs = [list(map(lambda x: x - best[i], y[i])) for i in range(len(best))]
        sample_calls = [a[2] for a in data]
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

        ax[0].set_ylabel("energy diffs")

        #fig.set_title("error bar")
        ax[0].errorbar(x, av, yerr=err, fmt='o', capsize=5, markersize=cs, color=color(maxiter), label='{} maxiter'.format(maxiter))
      
        ax[m+1].set_title("measurements")
        ax[m+1].set_ylabel("sampling calls")
        sample_calls_avgs = [sum(a)/len(a) for a in sample_calls]
        ax[m+1].bar(x, sample_calls_avgs)  # will be used to label x-ticks

        #plt.plot(x,av,label="av ({})".format(samples))
#plt.legend()
plt.xlabel("optimizer method")
plt.ylabel("finite sample energy")
plt.savefig("av_and_range_n.png")
plt.show()
exit(0)
fig, ax = plt.subplots(len(energy_diff_results), sharex=True, sharey=True)
for i in range(len(energy_diff_results)):
    ax[i].boxplot(energy_diff_results[i], patch_artist=True,  # fill with color
                   tick_labels=x)  # will be used to label x-ticks
    ax[i].set_title("box plot for {} shots".format(sample_list[i]))
    ax[i].set_ylabel("energy diff")
plt.savefig("box_plot.png")
plt.show()


