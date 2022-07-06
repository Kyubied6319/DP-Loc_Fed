import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pylab import *
import csv
import itertools
import pickle
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
## in order to plot the results in the paper, you must set a few things:
# 1: at line 50 the X and the Y axis
# 2: at line 65 the cell size 
# 3: at line 68 the output filename
# 4:  at line 120 the results file (pickle files)
# 5: at line 126 the IN_FILENAME 
# 6:  at line 130 the labels

def plot_dist(name, data):
    plt.clf()
    # CDF
    # sort the data:
    data_sorted = np.sort(data)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    
    # plot the sorted data:
    axs[0].hist(data, bins=range(50), density=True)
    axs[0].set_title('Occurence per size')
    axs[0].set_xlabel('length')
    axs[0].set_xlim(2, 50)
    axs[1].plot(data_sorted, p)
    #axs[1].set_ylabel('$p$')
    axs[1].set_title('CDF')
    axs[1].set_xlabel('length')
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, 50)
    axs[1].yaxis.set_ticks(np.arange(0,1,1))
    axs[1].locator_params(nbins=50, axis='y')
    
    plt.savefig("dist_%s.png" % name)

# --- PARAMETERS TO SET ---

X_LIM = [0, 24]
#Y_LIM = [0, 1]
Y_LIM = [0, 5500]
#Y_LIM = [0,4500]
GRID_Y = True
GRID_X = False
LEGEND_LOC = 1

FONT_SIZE_XLABEL = 12
FONT_SIZE_YLABEL = 12
FONT_SIZE_XTICK = 12
FONT_SIZE_YTICK = 12
FONT_SIZE_LEGEND = 10
FONT_SIZE_MARKER = 10

#CELL_SIZE = 632
CELL_SIZE = 315
m = {315: 250, 632: 500}
OUT_FILENAME = "emd_source-dest_cell_%d_SF.pdf" % m[CELL_SIZE]
#OUT_FILENAME = "emd_density_cell_%d_SF.pdf" % m[CELL_SIZE]
#OUT_FILENAME = "jsd_%d_SF.pdf" % m[CELL_SIZE]

# --- END OF PARAMETERS ---

def marker_cycle():
   """ 
   Return an infinite, cycling iterator over the available marker symbols.
   This is wrapped in a function to make sure that you get a new iterator
   that starts at the beginning every time you request one.
   """
   return itertools.cycle([
        's','v', '^','d','D',
        'o','x', '+','<','>',
        '1','2','3','4','h',
        'H','p','|','_'])

def pattern_cycle():
    return itertools.cycle([ '-','--','-.',':'])

def color_cycle():
    #return itertools.cycle([str(i) for i in np.arange(0,1,0.2)])
   return itertools.cycle([ 'r','b', 'purple','g', 'cyan', 'k'])

# --- END OF ITERATORS ---

# WE USE LATEX FONTS
plt.clf()
plt.rc('text')
plt.rc('font')
ax = plt.gca()

# SET FONT SIZE OF ON AXIS LABELS
ax.xaxis.label.set_fontsize(FONT_SIZE_XLABEL)
ax.yaxis.label.set_fontsize(FONT_SIZE_YLABEL)

# SET FONT SIZE  WE NEED GRID LINES?
ax.yaxis.grid(GRID_Y)
ax.xaxis.grid(GRID_X)

# SET FONT SIZE OF TICKS ON X AND Y AXIS
for item in ax.get_yticklabels():
    item.set_fontsize(FONT_SIZE_YTICK)

for item in ax.get_xticklabels():
    item.set_fontsize(FONT_SIZE_XTICK)

# SETTING AXIS INTERVALS
plt.xlim(*X_LIM)
plt.ylim(*Y_LIM)

[trip_size_jsd_per_hour4, fs_results4, density_per_hour4, trip_dist_per_hour4] = pickle.load(open("../out/results-cell_%d-eps_0.50.pickle" % CELL_SIZE, "rb"))
[trip_size_jsd_per_hour1, fs_results1, density_per_hour1, trip_dist_per_hour1] = pickle.load(open("../out/results-cell_%d-eps_1.00.pickle" % CELL_SIZE, "rb"))
[trip_size_jsd_per_hour2, fs_results2, density_per_hour2, trip_dist_per_hour2] = pickle.load(open("../out/results-cell_%d-eps_2.00.pickle" % CELL_SIZE, "rb"))
[trip_size_jsd_per_hour3, fs_results3, density_per_hour3, trip_dist_per_hour3] = pickle.load(open("../out/results-cell_%d-eps_5.00.pickle" % CELL_SIZE, "rb"))


#IN_FILENAMES = [(trip_size_jsd_per_hour4, r"$\varepsilon=0.5$"), (trip_size_jsd_per_hour1, r"$\varepsilon=1.0$"), (trip_size_jsd_per_hour2, r"$\varepsilon=2.0$"), (trip_size_jsd_per_hour3, r"$\varepsilon=5.0$")]
#IN_FILENAMES = [(density_per_hour4, r"$\varepsilon=0.5$"), (density_per_hour1, r"$\varepsilon=1.0$"), (density_per_hour2, r"$\varepsilon=2.0$"), (density_per_hour3, r"$\varepsilon=5.0$")]
IN_FILENAMES = [(trip_dist_per_hour4, r"$\varepsilon=0.5$"), (trip_dist_per_hour1, r"$\varepsilon=1.0$"), (trip_dist_per_hour2, r"$\varepsilon=2.0$"), (trip_dist_per_hour3, r"$\varepsilon=5.0$")]

x_label = "Hour"
#y_label = "JSD"
#y_label = "EMD Density"
y_label = "EMD Source-Destination"


# PLOT DATA
for (data, label), iter_marker, iter_color, iter_pattern in zip(IN_FILENAMES,
        marker_cycle(), color_cycle(), pattern_cycle()):
    #print(data) 
    #print(type(data))
    x = range(24)
    y = [data[i] for i in x]
    data_name = label

    # AXIS LABELS
    xlabel(x_label)
    ylabel(y_label)

    # HERE IS THE MARKER SIZE
    plot(x, y, color=iter_color, marker=iter_marker, linestyle=iter_pattern,
        markersize=FONT_SIZE_MARKER, label=r"%s" % data_name)

    # FONT SIZE OF LEGEND
    plt.legend(loc=LEGEND_LOC, prop={"size":FONT_SIZE_LEGEND})

plt.savefig(OUT_FILENAME)


