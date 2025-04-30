import csv


def csv_to_optim_data(filename):
    data = []
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data_row = [row[0], int(row[1]), int(row[2]), float(row[3])]
            energies = list(map(float, row[4:]))
            data_row.append(energies)
            data.append(data_row)
    return data

def csv_to_exp_data(filename):
    data = []
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data_row = [int(row[0]), float(row[1]), int(row[2]), float(row[3])]
            energies = list(map(float, row[4:]))
            data_row.append(energies)
            data.append(data_row)
    return data


if __name__ == "__main__":
    data = csv_to_optim_data("data/optim_data_200_10.csv")
    print(data)