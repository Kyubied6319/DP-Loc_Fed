from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import heapq
import sys
import time
from math import log, exp
from multiprocessing import Pool
import random
from operator import itemgetter

import networkx as nx
import tensorflow.compat.v1 as tf
import statistics as st
import numpy as np
from preproc import *
from utils import *
#from tqdm import tqdm
import os

np.set_printoptions(threshold=np.inf)

cnf = load_cfg("cfg/cfg_general.json")
cnf.CELL_SIZE = int(float(sys.argv[2]))
cnf.__dict__.update(load_cfg(sys.argv[3]).__dict__)

preproc = Preprocessing(cnf.CELL_SIZE, cnf.EPS)
#preproc.top_k_size = 234
nodes = preproc.cell2token.values()
projector = Projector(cnf.MIN_X, cnf.MIN_Y, cnf.MAX_X, cnf.MAX_Y)
nearests = {}
for n in nodes:
    temp = {}
    for x in nodes:
        if x != n:
            try:
                temp[x] = cell_dist(projector, preproc.id2cell[preproc.token2cell[n]], preproc.id2cell[preproc.token2cell[x]])
            except:
                pass
    nearests[n] = [k for k, v in heapq.nsmallest(20, temp.items(), key=lambda item: item[1])]
print("Preprocess and projector done")
def generate_init_data(vae, num):
    return vae.sample(num).numpy()


def process(args):
    start_time = time.time()
    (chunk, total_chunk, data) = args
    print("Processing chunk %d/%d..." % (chunk, total_chunk))

    from models import create_model_nh_ffn

    if cnf.PREPROC_MAP_TO_TOP_K:
        model = create_model_nh_ffn(max_words_num=len(preproc.token2cell.values()), out_size=preproc.top_k_size)
    else:
        model = create_model_nh_ffn(max_words_num=len(preproc.token2cell.values()), out_size=preproc.nl_size)

    rep_file = cnf.PATH_NH % (cnf.CELL_SIZE, cnf.EPS)
    model.load_weights(rep_file)

    def build_graph(dst, slot):
        G = nx.DiGraph()

        nodes = preproc.cell2token.values()

        inp = np.float32([[[node, dst, slot] for node in nodes]])
        #inp = inp.reshape(len(inp[0], 1, 3))
        inp = inp.reshape(len(inp[0]), 1, 3)
        outputs = tf.nn.softmax(model(inp)).numpy()

        for i, node in enumerate(nodes):
            output = outputs[i]
            try:
                curr_loc = preproc.id2cell[preproc.token2cell[node]]
            except KeyError:
                continue

            edges = []
            for j, prob in enumerate(output):
                if cnf.PREPROC_MAP_TO_TOP_K:
                    tmp = preproc.inv_tm[j]
                else:
                    tmp = preproc.inv_nm[j]
                    tmp = (curr_loc[0] + tmp[0], curr_loc[1] + tmp[1])
                try:
                    edges.append((node, preproc.cell2token[preproc.cell2id[tmp]], -log(prob)))
                except (KeyError, ValueError) as e:
                    try:
                        edges.append((node, preproc.cell2token[preproc.cell2id[tmp]], -log(float(0.0000000001))))
                    except (KeyError, ValueError) as e:
                        print("error: ", e)
                        print("node1: ", node)
                        print("node2: ", preproc.cell2token[preproc.cell2id[tmp]])
                        print("prob: ",prob)
                        pass
            G.add_weighted_edges_from(edges)
        return G
    
    count_new_traces = 0
    traces0 = []
    traces1 = []
    traces5 = []
    traces10 = []
    traces10 = []
    traces25 = []
    traces50 = []
    traces100 = []
    traces150 = []
    count2s = 0
    print("traces")
    counting = 0
    for key_dt in data:
        dst = key_dt[0]
        ts = key_dt[1]
        G = build_graph(dst=dst, slot=ts)
        le_set = set(data[key_dt])
        for src in le_set:
            if src == dst:
                continue
            try:
                trace_len, trace = nx.bidirectional_dijkstra(G, source=src, target=dst, weight='weight')
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                print("line 122")
                continue
            # select a uniformly random node for new route for Monte Carlo alg
            for rep in range(0, data[key_dt].count(src)):
                if len(trace)<3:
                    count2s += 1
                    loop_trace = []
                    for t in trace:
                        loop_trace.append(t)
                        try:
                            count = max(0, min(np.random.geometric(1-(exp(-G[t][t]["weight"]))),3))
                        except:
                             count = max(0, min(np.random.geometric(0.1), 3))
                        loop = [t]*count
                        loop_trace.extend(loop) 
                    traces0.append([dst, ts, loop_trace])
                    traces1.append([dst, ts, loop_trace])
                    traces5.append([dst, ts, loop_trace])
                    traces10.append([dst, ts, loop_trace])
                    continue
                else:
                    reject = 0
                    original_trace = trace
                    for iters in [0, 10]:
                        if iters == 0:
                            for rep in range(0, data[key_dt].count(src)):
                                loop_trace = []
                                for t in trace:
                                    try:
                                        count = max(0, min(np.random.geometric(1 - (exp(-G[t][t]["weight"]))), 3))
                                    except:
                                        count = max(0, min(np.random.geometric(0.1), 3))
                                    loop = [t] * count
                                    loop_trace.append(t)
                                    loop_trace.extend(loop)
                    
                                traces0.append([dst, ts, loop_trace])
                        else:
                            for MCMC in range(1, 11):
                                if reject == 1:
                                    reject = 0
                                    trace = old_trace
                                elif MCMC == 1:
                                    pass
                                else:
                                    trace = new_trace
        
                                x = dst
                                while x == dst or x == trace[-2]:
                                    x = random.choice(trace)
                                neighbours = [n for n in G.neighbors(x)]
                                #get close topk points:
                                candidate_neighbour = random.choice(nearests[x])
                                #candidate_neighbour = random.choice(neighbours)
                                cand_weight1 = G[x][candidate_neighbour]["weight"]
                                try:
                                    xn = trace[trace.index(x) + 2]
                                except:
                                    print(trace, original_trace)
        
                                cand_weight2 = G[candidate_neighbour][xn]["weight"]
                                old_weight1 = G[x][trace.index(x) + 1]["weight"]
                                old_weight2 = G[trace.index(x) + 1][xn]["weight"]
                                #new_trace_len = trace_len + cand_weight1 + cand_weight2 - old_weight1-old_weight2
                                new_trace_len = cand_weight1 + cand_weight2
                                trace_len = old_weight1+old_weight2
                                new_trace = []
                                for g in range(len(trace)):
                                    #here we skip one point in the old trace, so we use x instead
                                    if g != trace.index(x)+1:
                                        new_trace.append(trace[g])
                                    if g == trace.index(x)+1:
                                        new_trace.append(candidate_neighbour)
                                        
                                # now calculate the probabilities of each route
                                p_trace = exp(-trace_len)
                                p_new = exp(-new_trace_len)
        
                                # generate random alpha for metropolis hastings:
                                alpha = random.random()
                                candidate = p_new / p_trace
                                if alpha > candidate:
                                    # reject cand
                                    reject = 1
                                    old_trace = trace
                                    loop_trace = []
                                    for t in trace:
                                        loop_trace.append(t)
                                        try:
                                            count = max(0, min(np.random.geometric(1-(exp(-G[t][t]["weight"]))),3))
                                        except Exception as e:
                                            count = max(0, min(np.random.geometric(0.1), 3))
                                        loop = [t]*count
                                        loop_trace.extend(loop)

                                else:
                                    reject = 0
                                    loop_trace = []
                                    for t in new_trace:
                                        loop_trace.append(t)
                                        try:
                                            count = max(0, min(np.random.geometric(1-(exp(-G[t][t]["weight"]))), 3))
                                        except Exception as e:
                                            count = max(0, min(np.random.geometric(0.1),3))
                                        loop = [t]*count
                                        loop_trace.extend(loop)
                                if MCMC == 1:
                                    traces1.append([dst, ts, loop_trace])
                                elif MCMC == 5:
                                    traces5.append([dst, ts, loop_trace])
                                elif MCMC == 10:
                                    counting += 1
                                    #print(loop_trace)
                                    traces10.append([dst, ts, loop_trace])
                                elif MCMC == 25:
                                    traces25.append([dst, ts, loop_trace])
                                elif MCMC == 50:
                                    traces50.append([dst, ts, loop_trace])
                                elif MCMC == 100:
                                    traces100.append([dst, ts, loop_trace])
                                elif MCMC == 150:
                                    traces150.append([dst, ts, loop_trace])


    #Uncomented
    
    print("non 2 longs: ", counting)
    print("count2s: ", count2s)
    traces = [traces0, traces1, traces5, traces10, traces25, traces50, traces100, traces150]

    #syn_trace_lens = [len(i[2]) for trace in traces for i in trace ]
    #print("mean syn lens:", st.mean(syn_trace_lens), "median: ", st.median(syn_trace_lens), "mode: ",st.mode(syn_trace_lens))
    #print("stat for repetitions: mean: ", st.mean(count2s), "median: ", st.median(count2s), "mode: ", st.mode(count2s), "max: ", max(count2s), "min: ", min(count2s), "stdev: ", st.stdev(count2s) )
    #print("node not found: ", no_path)
    print("Processing of chunk %d/%d finished in %.3f seconds." % (chunk, total_chunk, time.time() - start_time))
    print("new traces: ", count_new_traces)
    
    return traces


if __name__ == '__main__':
    rep_init_file = cnf.GENERATED_INIT_DATA_FILE % (cnf.CELL_SIZE, cnf.EPS)

    # Note if you import (tensorflow) here then multiprocessing won't work, hence this stupid solution...
    if sys.argv[1] == "VAE":
        from models import VAE
        print("--------------------------------------- INIT GENERATOR --------------------------------------------------")

        t1 = time.time()
        vae = VAE(original_dim=cnf.VAE_ORIG_DIM, latent_dim=cnf.VAE_LATENT_DIM, hidden_dim=cnf.VAE_HIDDEN_DIM,
                  max_words_num=len(preproc.token2cell.values()), max_slots=preproc.MAX_SLOTS)
        rep_file = cnf.PATH_VAE % (cnf.CELL_SIZE, cnf.EPS)
        vae.load_weights(rep_file)
        print("vae model loaded")
        data = generate_init_data(vae, num=preproc.train_size)

        print("Data records before filtering:", len(data))

        def filter(row):
            return row[0] != 0 and row[1] != 0

        data = data[np.array([filter(row) for row in data])]

        print("Data records after filtering:", len(data))

        # Dump data for evaluation
        pickle.dump(data, open(rep_init_file, "wb"))
        print("Finished Init", time.time()-t1)
    else:
        print("---------------------------------------TRACE  GENERATOR --------------------------------------------------")
        t2 = time.time()
        data = pickle.load(open(rep_init_file, "rb"))

        data_vis = preproc.convert_init_data_to_coords(data)

        preproc.plot_init_data("vae_DP", data_vis)
  
        pool = Pool(cnf.CPU_CORES)
        # here we group the sources according to time-dest, so that fewer graphs are needed in general:
        sdt_dict = {}
        sdt_cnt = {}
        listsdt = []
        for i in range(0, cnf.CPU_CORES):
            listsdt.append({})
        for i, (src, dest, time) in enumerate(data):
            if (dest, time) in listsdt[i % cnf.CPU_CORES]:
                listsdt[i % cnf.CPU_CORES][(dest, time)].append(src)
            else:
                listsdt[i % cnf.CPU_CORES][(dest, time)] = []
                listsdt[i % cnf.CPU_CORES][(dest, time)].append(src)
        print("line 312")
        data_chunks = [(i + 1, cnf.CPU_CORES, listsdt[i]) for i in range(len(listsdt))]
        print("line 314")
        traces = pool.map(process, data_chunks)
        pool.close()
        pool.join()
        print("line 318")
        pickle.dump(traces, open("Pout/traces.pickle", "wb"))
        print("line 320")
        mcmcs = [0, 1, 5, 10, 25, 50, 100, 150]
        for MCMC in mcmcs:
            print("line3123")
            rep_traces_file = cnf.GENERATED_TRACES_FILE % (cnf.CELL_SIZE, cnf.EPS, MCMC)
            tr = []
            for line in traces:
                for element in line[mcmcs.index(MCMC)]:
                    tr.append(element)
                #tr = [x for y in traces[mcmcs.index(MCMC)] for x in y]
            pickle.dump(tr, open(rep_traces_file, "wb"))
        print("Finish Vae: ", time.time()-t2)
