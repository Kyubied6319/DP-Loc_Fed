import pickle
import statistics as st
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import *
from mlxtend.preprocessing import TransactionEncoder
from pyemd import emd
from scipy.spatial.distance import jensenshannon
import random
from preproc import Preprocessing
from utils import load_cfg, diff_coords
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cnf = load_cfg("cfg/cfg_general.json")
cnf.CELL_SIZE = int(float(sys.argv[1]))
cnf.__dict__.update(load_cfg(sys.argv[2]).__dict__)
MCMC = int(sys.argv[3])

print("------------------------ EVALUATION WITH MCMC ITERS ", MCMC, "--------------------------------------------------")

preproc = Preprocessing(cnf.CELL_SIZE, cnf.EPS)


def fp(name, dataset):
    print("---> Computing frequent itemsets for %s <---" % name)
    print(len(dataset))
    # dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    #        ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    #        ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    #        ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    #        ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    # fs = apriori(df, min_support=0.01, use_colnames=True)
    fs = fpgrowth(df, min_support=0.05, use_colnames=True, max_len=7)
    return fs.sort_values(by=['support'], ascending=False).reset_index(drop=True)


def compute_dist_matrix(coords, distance):
    size = len(coords)
    mat = np.zeros((size, size))

    for i, coord1 in enumerate(coords):
        for j, coord2 in enumerate(coords):
            if i == j:
                continue
            # symmetrical matrix, we can copy the upper triangular part to the lower one
            elif j < i:
                mat[i, j] = mat[j, i]
            else:
                mat[i, j] = distance(coord1, coord2)
    return mat


def process_VAE(args):
    orig, syn, ts = args

    start_time = time.time()

    # Select all samples according to time
    src_orig = orig[orig[:, 2] == ts, :]
    src_syn = syn[syn[:, 2] == ts, :]

    # For speed, we use only a random subsample
    src_orig = src_orig[np.random.choice(src_orig.shape[0], min(src_orig.shape[0], cnf.EMD_MIN_SAMPLES), replace=False),
               :]
    src_syn = src_syn[np.random.choice(src_syn.shape[0], min(src_syn.shape[0], cnf.EMD_MIN_SAMPLES), replace=False), :]

    s_cnt = [(src, dst) for (src, dst, _) in src_syn]
    o_cnt = [(src, dst) for (src, dst, _) in src_orig]
    cnt = Counter(s_cnt + o_cnt)
    rows = cnt.keys()

    #print("Domain size in slot %d: %d" % (ts, len(rows)))

    coords = [(preproc.id2cell[preproc.token2cell[x]], preproc.id2cell[preproc.token2cell[y]]) for (x, y) in rows]
    distance = lambda x, y: np.linalg.norm(preproc.cell_size * np.array(diff_coords(x[0], y[0])), ord=2) + \
                            np.linalg.norm(preproc.cell_size * np.array(diff_coords(x[1], y[1])), ord=2)
    dist_m = compute_dist_matrix(coords, distance)

    s_dist = np.zeros(len(rows))
    s_cnt = Counter(s_cnt)
    for i, k in enumerate(rows):
        s_dist[i] = s_cnt[k]

    o_dist = np.zeros(len(rows))
    o_cnt = Counter(o_cnt)
    for i, k in enumerate(rows):
        o_dist[i] = o_cnt[k]

    emd_val = emd(s_dist / s_dist.sum(), o_dist / o_dist.sum(), dist_m)
    #print("VAE EMD in slot %d: %.2f meters" % (ts, emd_val), "Finished in %.3f seconds" % (time.time() - start_time))

    return emd_val


def comp_VAE(syn, orig, max_slots):
    params = [(orig, syn, ts) for ts in range(max_slots)]

    pool = Pool(cnf.CPU_CORES)

    emd_vals = pool.map(process_VAE, params)

    return emd_vals


def process_density(args):
    orig, syn, ts = args

    start_time = time.time()

    orig_traces = [trace for _, slot, trace in orig if slot == ts]
    syn_traces = [trace for _, slot, trace in syn if slot == ts]

    # For speed, we use only a random subsample
    #orig_traces = orig_traces[np.random.choice(orig_traces.shape[0], min(orig_traces.shape[0], EMD_MIN_SAMPLES), replace=False), :]
    #syn_traces = syn_traces[np.random.choice(syn_traces.shape[0], min(syn_traces.shape[0], EMD_MIN_SAMPLES), replace=False), :]
    orig_traces = random.sample(orig_traces, k=min(len(orig_traces), cnf.EMD_MIN_SAMPLES))
    syn_traces = random.sample(syn_traces, k=min(len(syn_traces), cnf.EMD_MIN_SAMPLES))

    s_cnt = [x for trace in orig_traces for x in trace]
    o_cnt = [x for trace in syn_traces for x in trace]
    cnt = Counter(s_cnt + o_cnt)
    rows = cnt.keys()

    #print("Domain size in slot %d: %d" % (ts, len(rows)))

    coords = [preproc.id2cell[preproc.token2cell[x]] for x in rows]
    distance = lambda x, y: np.linalg.norm(preproc.cell_size * np.array(diff_coords(x, y)), ord=2)
    dist_m = compute_dist_matrix(coords, distance)

    # Building histograms (pdf) with common domain
    s_dist = np.zeros(len(rows))
    s_cnt = Counter(s_cnt)
    for i, k in enumerate(rows):
        s_dist[i] = s_cnt[k]

    o_dist = np.zeros(len(rows))
    o_cnt = Counter(o_cnt)
    for i, k in enumerate(rows):
        o_dist[i] = o_cnt[k]

    emd_val = emd(s_dist / s_dist.sum(), o_dist / o_dist.sum(), dist_m)
    #print("Density EMD in slot %d: %.2f meters" % (ts, emd_val),
          #"Finished in %.3f seconds" % (time.time() - start_time))

    return emd_val


def comp_density(syn, orig, max_slots):
    params = [(orig, syn, ts) for ts in range(max_slots)]

    pool = Pool(cnf.CPU_CORES)

    emd_vals = pool.map(process_density, params)

    return emd_vals


def comp_fp(orig_data, syn_data):
    fs_orig = fp("orig", orig_data)
    fs_syn = fp("synthetic", syn_data)

    fs_orig = fs_orig['itemsets'].tolist()
    fs_syn = fs_syn['itemsets'].tolist()

    matching = {}
    for topK in cnf.TOP_K:
        fs_o = set(fs_orig[:topK])
        #print("original: ",fs_o)
        fs_s = set(fs_syn[:topK])
        #print("syn:", fs_s)
        if len(fs_s) == 0:
            continue
        matching[topK] = len(fs_o & fs_s) / len(fs_s)
        print("-> Itemsets with size %d" % topK)
        print("Matching:", matching[topK])

    return matching


def comp_lens(orig_trace_lens, syn_trace_lens):
    max_len = max(orig_trace_lens + syn_trace_lens) + 1
    orig_s = np.zeros(max_len)
    syn_s = np.zeros(max_len)

    unique, counts = np.unique(orig_trace_lens, return_counts=True)
    if len(unique) == 0:
        return 1
    orig_s[unique] = counts
    unique, counts = np.unique(syn_trace_lens, return_counts=True)
    if len(unique) == 0:
        return 1
    syn_s[unique] = counts

    return jensenshannon(orig_s, syn_s, 2)

def process_emd_traces(pair, syn, orig):

    start_time = time.time()
    #print(len(orig), len(syn)) 

    orig_traces = random.sample(orig, k=min(len(orig), cnf.EMD_MIN_SAMPLES))
    syn_traces = random.sample(syn, k=min(len(syn), cnf.EMD_MIN_SAMPLES))

    s_cnt = [x for trace in orig_traces for x in trace]
    o_cnt = [x for trace in syn_traces for x in trace]
    cnt = Counter(s_cnt + o_cnt)
    rows = cnt.keys()

    coords = [preproc.id2cell[preproc.token2cell[x]] for x in rows]
    distance = lambda x, y: np.linalg.norm(preproc.cell_size * np.array(diff_coords(x, y)), ord=2)
    dist_m = compute_dist_matrix(coords, distance)

    # Building histograms (pdf) with common domain
    s_dist = np.zeros(len(rows))
    s_cnt = Counter(s_cnt)
    for i, k in enumerate(rows):
        s_dist[i] = s_cnt[k]

    o_dist = np.zeros(len(rows))
    o_cnt = Counter(o_cnt)
    for i, k in enumerate(rows):
        o_dist[i] = o_cnt[k]

    emd_val = emd(s_dist / s_dist.sum(), o_dist / o_dist.sum(), dist_m)
    #print("trace EMD in slot %d: %.2f meters" % (pair[0], emd_val),
    # "Finished in %.3f seconds" % (time.time() - start_time))

    return emd_val


#dictonary where key is src-dst and value is the different routes inbetween
def get_dict(in_traces):
    traces = [trace for _, slot, trace in in_traces]
    sd_dict = {}
    for tr in traces:
        sd = (tr[0], tr[-1])
        if sd in sd_dict:
            sd_dict[sd].append(tr[1:-1])
        else:
            sd_dict[sd] = [tr[1:-1]]
    return sd_dict

def comp_trip(syn_traces, orig_traces):
    syn_dict = get_dict(syn_traces)
    orig_dict = get_dict(orig_traces)
    emds = []
    set_syn = []
    set_orig = []
    isnan = 0
    intersection = syn_dict.keys() & orig_dict.keys()
    for pair in syn_dict:
        if pair in orig_dict:
            tem = process_emd_traces(pair, syn_dict[pair], orig_dict[pair])
            if np.isnan(tem):
                isnan +=1
                continue
            else:
                emds.append(tem)
            #set_syn.append(len(set(syn_dict[pair])))
            #set_orig.append(len(set(orig_dict[pair])))
        else:
            continue
    print("trace emd returned nan for this many pairs: ", isnan, " where all orig pairs: ", len(orig_dict), "all syn pairs: ", len(syn_dict), "intersecion: ", len(intersection))

    return emds


# def main(orig_data, syn_data):
def main():
    rep_orig_traces_file = cnf.ORIG_TRACES_FILE % cnf.CELL_SIZE
    rep_syn_traces_file = cnf.GENERATED_TRACES_FILE % (cnf.CELL_SIZE, cnf.EPS, MCMC)
    #rep_syn_traces_file = "Gout/gen.pickle"
    
    # Length of trips
    orig_traces = pickle.load(open(rep_orig_traces_file, "rb"))
    syn_traces = pickle.load(open(rep_syn_traces_file, "rb"))
    
    # syn_traces = pickle.load(open("syn_ngram.pickle", "rb"))
    trip_size_jsd_per_hour = {}
    for ts in range(preproc.MAX_SLOTS):
        orig_trace_lens = [len(trace) for _, slot, trace in orig_traces if slot == ts]
        syn_trace_lens = [len(trace) for _, slot, trace in syn_traces if slot == ts]

        trip_size_jsd_per_hour[ts] = comp_lens(orig_trace_lens, syn_trace_lens)
        print("JSD in slot %d: %.3f" % (ts, trip_size_jsd_per_hour[ts]))

    orig_trace_lens = [len(trace) for _, slot, trace in orig_traces]
    syn_trace_lens = [len(trace) for _, slot, trace in syn_traces]

    trip_size_jsd_per_hour['overall'] = comp_lens(orig_trace_lens, syn_trace_lens)

    # Frequent patterns
    fs_results = {}
    """
    for ts in range(preproc.MAX_SLOTS):
        orig_data = [trace for _, slot, trace in orig_traces if ts == slot]
        syn_data = [trace for _, slot, trace in syn_traces if ts == slot]

        fs_results[ts] = comp_fp(orig_data, syn_data)
    """
    orig_data = [trace for _, _, trace in orig_traces]
    syn_data = [trace for _, _, trace in syn_traces]

    fs_results['overall'] = comp_fp(orig_data, syn_data)

    # Density
    density_per_hour = comp_density(syn_traces, orig_traces, preproc.MAX_SLOTS)

    # Trip distribution
    rep_syn_init_file = cnf.GENERATED_INIT_DATA_FILE % (cnf.CELL_SIZE, cnf.EPS)
    rep_orig_init_file = cnf.ORIG_INIT_DATA_FILE % cnf.CELL_SIZE
    syn_init_data = np.int16(pickle.load(open(rep_syn_init_file, "rb")))
    orig_init_data = np.int16(pickle.load(open(rep_orig_init_file, "rb")))

    trip_dist_per_hour = comp_VAE(syn_init_data, orig_init_data, preproc.MAX_SLOTS)
    
    # EMD between src and dst:
    trace_dist = comp_trip(syn_traces, orig_traces)
    print("avg trace dist between src and dst: ", st.mean(trace_dist))
    #print("average number of diff cells in syn: ", st.mean(lsyn))
    #print("average number of diff cells in orig: ", st.mean(lorig))


    rep_file = cnf.RESULTS_FILE % (cnf.CELL_SIZE, cnf.EPS)
    pickle.dump([trip_size_jsd_per_hour, fs_results, density_per_hour, trip_dist_per_hour], open(rep_file, "wb"))

    print("Overall JSD: ", trip_size_jsd_per_hour['overall'])
    vae_avg = st.mean(trip_dist_per_hour)
    print("Mean VAE Density: ", vae_avg)
    dens_avg = st.mean(density_per_hour)
    print("Mean Density: ", dens_avg)
    print("Frequent Patterns Matching overall: ", fs_results['overall'])
    print("mean orig lens:", st.mean(orig_trace_lens), "median: ", st.median(orig_trace_lens), "mode: ",
            st.mode(orig_trace_lens), "stdev: ", st.stdev(orig_trace_lens))
    print("mean syn lens:", st.mean(syn_trace_lens), "median: ", st.median(syn_trace_lens), "mode: ",
          st.mode(syn_trace_lens), "stdev: ", st.stdev(syn_trace_lens))


if __name__ == '__main__':
    main()
