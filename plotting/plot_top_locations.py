import pickle

import matplotlib.pyplot as plt
import numpy as np

import utils

plt.style.use("ggplot")

cnf = utils.load_cfg("cfg/cfg_general.json")


def plot_data(k, step):
    with open(cnf.LOCATION_FREQUENCY_FILE % cnf.TAXI_NUM, "rb") as f:
        freq = np.array(pickle.load(f))

    top_k = np.array(freq[:k]) if k is not None else freq
    top_k[:, 0] = range(len(top_k))

    plt.figure(figsize=(18, 10))
    n, _, _ = plt.hist(top_k[:, 0], bins=np.arange(0, len(top_k) + 1, step), weights=top_k[:, 1] / np.sum(freq[:, 1]))
    plt.xticks(np.arange(0, len(top_k) + 1, step))
    plt.title(f"Top location visit distribution, {np.sum(n) * 100}% of all visits")
    plt.savefig(
        f"top-{k}-locations-{cnf.TAXI_NUM}-GEO_norep.png" if k is not None else "top-locations-{cnf.TAXI_NUM}-taxis.png")


if __name__ == "__main__":
    plot_data(None, 100)
    plot_data(1000, 50)
    plot_data(500, 20)
    plot_data(200, 20)
    plot_data(100, 10)
