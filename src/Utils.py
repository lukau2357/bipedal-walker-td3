import matplotlib.pyplot as plt
import csv
import os
import numpy as np

MAIN_ENV = "BipedalWalker-v3"

"""
Plotting utilities
"""

def plot_ep_history(model_suffix = "", dir = "", version = 0):
    path = "{}{}_{}-data.csv".format(MAIN_ENV, model_suffix, str(version))
    path = os.path.join(dir, path)
    scores = []

    with open(path, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) == 0:
                continue

            scores.append(float(row[1]))
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.set_title("Model {} history".format(str(version)))
    ax.plot(scores)
    ax.set_ylabel("Score achieved")
    ax.set_xlabel("Episode number")
    plt.show()


def plot_ci(model_suffix = "", dir = "history"):
    path = os.path.join(".", dir)
    data = []

    for file in os.listdir(path):
        if model_suffix not in file:
            continue

        current_scores = []

        with open(os.path.join(path, file), "r") as f:
            reader = csv.reader(f, delimiter = ",")

            for row in reader:
                if len(row) == 0:
                    continue
                
                current_scores.append(float(row[1]))
        
        data.append(current_scores)

    data = np.asarray(data)
    mu = data.mean(axis = 0)
    std = data.std(axis = 0) / np.sqrt(data.shape[0])
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.plot([i for i in range(1, data.shape[1] + 1)], mu, label = "Average reward obtained")
    ax.fill_between(list(range(1, data.shape[1] + 1)), 
                    mu - 1.96 * std, mu + 1.96 * std, alpha = 0.1, color = "b", 
                    label = r"95% confidence region")
    ax.axhline(y = 0, linestyle = "--", color = "black")
    xticks = [i for i in range(1, data.shape[1] + 1) if i % 100 == 0 ]
    xticks.insert(0, 1)
    ax.set_xticks(xticks)
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Score")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    plot_ci()