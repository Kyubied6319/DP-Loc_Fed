from math import radians, cos, sin, asin, sqrt, ceil, log2 
from mpl_toolkits.basemap import Basemap
from os import listdir, stat
import os
from os.path import join, isfile, getsize
import numpy as np
import tensorflow.compat.v1 as tf
from collections import Counter

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import colors
from operator import itemgetter
import json
from types import SimpleNamespace 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
plt.style.use('ggplot')

diff_coords = lambda x,y : (x[0] -y[0], x[1] -y[1])
sum_coords = lambda x,y : (x[0] + y[0], x[1] + y[1])

# expects: [(x1,c1), (x2, c2), ...], where xi is the item, and ci is its occurence
def percentile(data, percent = 0.8):
    s = sum([x[1] for x in data])
    run_sum = 0
    for i, x in enumerate(sorted(data, key=itemgetter(1), reverse=True)):
        run_sum += x[1]
        if run_sum/s >= percent: 
            break
    return i

def load_cfg(fname):
    cfg = json.load(open(fname))
    return SimpleNamespace(**cfg)

@tf.function
def tf_inv_sigmoid(x):
  """ Computes the logit function, i.e. the logistic sigmoid inverse. """
  return - tf.log(1. / x - 1.)
  
def np_inv_sigmoid(x, T = 1):
  """ Computes the logit function, i.e. the logistic sigmoid inverse. """
  return - np.log(1. / x - 1.) / T

def plot_init_data(name, data, max_x, max_y, max_t, cell_size):
    fig, axs = plt.subplots(1, 3, figsize=(15,4))

    def plot_cmap(ax, name, cnt, colorb = False):
        vis_data = np.zeros((max_x, max_y))
        for coord in cnt: 
            #print (max_y, coord[1])
            c = list(coord)
            c[0] = min(max_x, c[0])
            c[0] = max(1, c[0])
            c[1] = min(max_y, c[1])
            c[1] = max(1, c[1])
            vis_data[max_x - c[0], max_y - c[1]] = cnt[coord]
            
        cs = ax.pcolor(vis_data[::-1], cmap=plt.cm.jet, vmin=0, vmax=max(cnt.values()), edgecolors='k', linewidths=0.1)
        if colorb:
            cbar = plt.colorbar(cs, ax=[ax], location='left')
        ax.set_title(name)

    # Source
    data = np.array(data, dtype=int)
    cnt = Counter([tuple(x) for x in data[:, 0:2]])
    plot_cmap(axs[0], name + "_source", cnt, colorb=True)

    # Destination
    cnt = Counter([tuple(x) for x in data[:, 2:4]])
    plot_cmap(axs[1], name + "_dest", cnt)
    
    # Time
    axs[2].hist(data[:, 4], bins=max_t)
    axs[2].set_title(name + "_start_time")

    fig.suptitle("Population: %d, Cell size: %.2f meters" % (data.shape[0], cell_size))
    
    plt.savefig("init_data_%s.png" % name) 
    plt.clf()
    plt.close()

'''
def get_neighbors(loc):
    cl = dict()

    def addNeighbor(seq, loc):
        if loc[0] > 0 and loc[1] > 0:
            cl[seq] = loc 
            
    # we add the node itself
    addNeighbor(0, (loc[0], loc[1]))

    # neighboring cells by side
    addNeighbor(1, (loc[0] - 1, loc[1]))
    addNeighbor(2, (loc[0] + 1, loc[1]))
    addNeighbor(3, (loc[0], loc[1] - 1))
    addNeighbor(4, (loc[0], loc[1] + 1))
    
    # neighboring cells by corners
    addNeighbor(5, (loc[0] - 1, loc[1] + 1))
    addNeighbor(6, (loc[0] - 1, loc[1] - 1))
    addNeighbor(7, (loc[0] + 1, loc[1] - 1))
    addNeighbor(8, (loc[0] + 1, loc[1] + 1))

    return cl
'''

def get_neighbors(hops):
    nl = set([(0,0)])
    for i in range(1, hops + 1):
        for j in range(-i, i + 1):
            nl.add((i, j))
            nl.add((-i, j))
            nl.add((j, i))
            nl.add((j, -i))

    return nl

def get_neighbor_code(loc_src, loc_nb, neighbor_map):
    #return get_neighbors(loc_src)[loc_nb]
    return neighbor_map[(loc_src[0] - loc_nb[0], loc_src[1] - loc_nb[1])]
    
# For stat printing
def print_stat(name, v):
    print ("---> %s <---" % name)
    print ("Total:", len(v)) 
    print ("mean: %.2f, median: %d, max: %d, min: %d, std.dev: %.2f" % (np.mean(v), np.median(v), max(v), min(v), np.std(v)))

# http://blog.notdot.net/2009/11/Damn-Cool-Algorithms-Spatial-indexing-with-Quadtrees-and-Hilbert-Curves
hilbert_map = {
    'a': {(0, 0): (0, 'd'), (0, 1): (1, 'a'), (1, 0): (3, 'b'), (1, 1): (2, 'a')},
    'b': {(0, 0): (2, 'b'), (0, 1): (1, 'b'), (1, 0): (3, 'a'), (1, 1): (0, 'c')},
    'c': {(0, 0): (2, 'c'), (0, 1): (3, 'd'), (1, 0): (1, 'c'), (1, 1): (0, 'b')},
    'd': {(0, 0): (0, 'a'), (0, 1): (3, 'c'), (1, 0): (1, 'd'), (1, 1): (2, 'd')},
}

# Locality preserving mapping from 2D to 1D with Hilbert-curve
# An order 1 curve fills a 2x2 grid, an order 2 curve fills a 4x4 grid, and so forth

def point_to_hilbert_curve(x, y, order=16):
    x = int(round(x))
    y = int(round(y))
    current_square = 'a'
    position = 0
    for i in range(order - 1, -1, -1):
        position <<= 2
        quad_x = 1 if x & (1 << i) else 0
        quad_y = 1 if y & (1 << i) else 0
        quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
        position |= quad_position
    return position

# geo distance between two gps locations
def haversine(gps1, gps2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    (lon1, lat1) = gps1
    (lon2, lat2) = gps2

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km

file_len = lambda name : os.stat(name).st_size

def sort_files(dirname, desc=True, fnames_only=True):
    """ Returns with list of filenames of directory, ordered by file sizes.
    For instance:
    dirname = D, desc = True, fnames_only=True

    D--
       |_a.txt (56 kB) -> getsize()  | or (56 rows) -> file_len()
       |_b.txt (114 kB)              | or (114 rows)
       |_c.txt ( 8 kB)               | or (8 rows)

       returns: ['b.txt', 'a.txt', 'c.txt']
       """
    files = [join(dirname, basename) for basename in listdir(dirname) if isfile(join(dirname, basename))]
    if fnames_only:
        return [t[0] for t in sorted([(fname, file_len(fname)) for fname in files], key=lambda t: t[1], reverse=desc)]
    else:
        return sorted([(fname, file_len(fname)) for fname in files], key=lambda t: t[1], reverse=desc)

class Projector:
    def __init__(self, min_x, min_y, max_x, max_y):
        '''
        llcrnrlon	longitude of lower left hand corner of the desired map domain (degrees).
        llcrnrlat	latitude of lower left hand corner of the desired map domain (degrees).
        urcrnrlon	longitude of upper right hand corner of the desired map domain (degrees).
        urcrnrlat	latitude of upper right hand corner of the desired map domain (degrees).
        '''
        # Use mercator projection by default
        self.m = Basemap(llcrnrlon=min_x,llcrnrlat=min_y,urcrnrlon=max_x,urcrnrlat=max_y, resolution='h',projection='merc')

    '''
    Helper function: Rather use toGPS or toProjected
    inv = True: projected -> GPS
    inv = False: GPS -> projected
    '''
    def project(self, x,y,inv=False):
        # Calling a Basemap class instance with the arguments lon, lat 
        # will convert lon/lat (in degrees) to x/y map projection coordinates
        # (in meters). The inverse transformation is done if the optional 
        # keyword inverse is set to True. 
        return self.m(x,y, inverse = inv)

    '''
    Project a single GPS coordinate
    x is lon and y is lat
    '''
    def toProjected(self, x,y):

        (p1, p2) = self.project([x],[y])

        return (p1[0],p2[0])

    '''
    Convert a single projected coordinate to lon/lat 
    coord[0] is lon and coord[1] is lat
    '''
    def toGPS(self, coord):

        (lon, lat) = self.project([coord[0]],[coord[1]],inv=True)

        return (lon[0],lat[0])

    '''
    Convert multiple projected coordinates to lons/lats 
    Input: [(x1,y1), (x2, y2), ..., (xn, yn)]
    Output: [(lon1,lat1), (lon2, lat2), ..., (lonn, latn)]
    '''
    def toGPS_list(self, coords):

        (x, y) = zip(*coords)

        # get lon/lat
        (lons, lats) = self.project(x,y,inv=True)

        return zip(lons, lats) 

    '''
    Project multiple GPS coordinates    
    Input: [(lon1,lat1), (lon2, lat2), ..., (lonn, latn)]
    Output: [(x1,y1), (x2, y2), ..., (xn, yn)]
    '''
    def toProjected_list(self, coords):

        (lons, lats) = zip(*coords)

        # get lon/lat
        (x, y) = self.project(lons,lats,inv=True)

        return zip(x, y) 

