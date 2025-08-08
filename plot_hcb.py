from csv_to_data import csv_to_hcb
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    data = csv_to_hcb("exp_sampling_results/hcb_analysis.csv")
    x = [a[1] for a in data]
    print(x)