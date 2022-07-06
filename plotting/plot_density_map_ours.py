import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pickle
from collections import Counter

plt.style.use('ggplot')

def plot_init_data(name, cnt, max_x, max_y, cell_size):
    #fig, axs = plt.subplots(1, 2, figsize=(15,4))
    fig, axs = plt.subplots()

    def plot_cmap(ax, cnt, max_val, colorb = False):
        vis_data = np.zeros((max_x, max_y))
        for coord in cnt: 
            #print (max_y, coord[1])
            c = list(coord)
            c[0] = min(max_x, c[0])
            c[0] = max(1, c[0])
            c[1] = min(max_y, c[1])
            c[1] = max(1, c[1])
            vis_data[max_x - c[0], max_y - c[1]] = cnt[coord]
            
        cs = ax.pcolor(vis_data[::-1], cmap=plt.cm.jet, vmin=0, vmax=max_val, edgecolors='k', linewidths=0.1)
        if colorb:
            cbar = plt.colorbar(cs, ax=[ax], location='left')

    plot_cmap(axs, cnt, max(cnt.values()), colorb=True)
    #plot_cmap(axs, cnt, 250000, colorb=True)

    #fig.suptitle("Visits: %d, Cell size: %.2f meters" % (sum(x[1] for x in cnt), cell_size))
    fig.suptitle("Visits: %d, Cell size: %.2f meters" % (sum(x for x in cnt.values()), cell_size))
    
    plt.savefig("Density_%s_%d_GEO_eps1.pdf" % (name, cell_size)) 
    plt.clf()
    plt.close()

if __name__ == "__main__":
    #CELL_SIZE = 315
    CELL_SIZE = 632
    # cell_size is in meters, CELL_SIZE is in mercator
    MAP_WIDTH, MAP_HEIGHT, MAX_SLOTS, cell2id, cell_size =\
        pickle.load(open("Gout/aux_cell_info_for_preproc-cell_%d.pickle" % CELL_SIZE,"rb"))
    id2cell = { v : k for k,v in cell2id.items()}
    
    [token2cell, nl_size, neighbor_map, train_size] =\
            pickle.load(open("Gout/aux_preproc_info-cell_%d.pickle" % CELL_SIZE,"rb"))
    
    #traces = pickle.load(open("Pout/original_traces-cell_%d.pickle" % CELL_SIZE, "rb"))
    #traces = pickle.load(open("Gout/generated_traces-cell_%d-eps_1.00.pickle" % CELL_SIZE, "rb"))
    #traces = pickle.load(open(results_ngram_632_SF_eps1-anon-traces.pickle, "rb"))
    #density = Counter(id2cell[token2cell[y]] for x in traces for y in x[2])

    plot_init_data("DP-LOC", density, MAP_HEIGHT, MAP_WIDTH, cell_size)

