import matplotlib.pyplot as plt
import pickle

with open("data/dist_data_200.dat", "rb") as file:
    data = pickle.load(file)

x = [a[1] for a in data]
y = [a[-1] for a in data]
best = [a[-2] for a in data]

lo = [min(a) for a in y]
hi = [max(a) for a in y]
av = [sum(a)/len(a) for a in y]


plt.plot(x,lo,label="lo")
plt.plot(x,hi,label="hi")
plt.plot(x,av,label="av")
plt.plot(x,best,label="baseline", color="black")
plt.xlabel("H-H distance")
plt.ylabel("energy (Eh)")
plt.legend()
plt.savefig("energies.png")
plt.show()

plt.figure()

lo = [lo[i] - best[i] for i in range(len(lo))]
hi = [hi[i] - best[i] for i in range(len(lo))]
av = [av[i] - best[i] for i in range(len(lo))]

plt.plot(x,lo,label="lo")
plt.plot(x,hi,label="hi")
plt.plot(x,av,label="av")
plt.xlabel("H-H distance")
plt.ylabel("finite sample error (Eh)")
plt.legend()
plt.savefig("av_range_d.png")
plt.show()


