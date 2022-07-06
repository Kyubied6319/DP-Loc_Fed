import os
import pickle
from collections import Counter
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from tables import *
from tqdm import tqdm
import statistics as st

import utils

cnf = utils.load_cfg("cfg/cfg_general.json")

# compute distance between cells (for error calculation)
def to_mapped_coord(x):
    return x * cnf.CELL_SIZE + (cnf.CELL_SIZE / 2)


def cell_dist(projector, c1, c2):
    x1, y1 = to_mapped_coord(c1[0]), to_mapped_coord(c1[1])
    x2, y2 = to_mapped_coord(c2[0]), to_mapped_coord(c2[1])

    # store in meters (not km)
    return int(1000 * utils.haversine(projector.toGPS((x1, y1)), projector.toGPS((x2, y2))))


def add_noise_to_hist(hist, scale):
    """Add Gaussian noise to all bins in histogram. Return bins in decreasing order."""
    ids = np.array([x[0] for x in hist])
    counts = np.array([x[1] for x in hist], dtype="f")
    np.random.seed(42)
    counts += np.random.normal(scale=scale, size=len(hist))
    return ids[counts.argsort()[::-1]]


# TODO: cell2token maps only the used cells. But for DP we would need unused cells too!!!

class Preprocessing:

    def __init__(self, p_CELL_SIZE, p_EPS):
        [self.MAP_WIDTH, self.MAP_HEIGHT, self.MAX_SLOTS, self.cell2id, self.cell_size] = \
            pickle.load(open(cnf.AUX_CELL_INFO_FILE % p_CELL_SIZE, "rb"))
        # Be careful: self.cell_size is in meters, whereas cnf.CELL_SIZE is in coordinates
        cnf.CELL_SIZE = p_CELL_SIZE
        cnf.EPS = p_EPS
        self.cell2token = None
        self.id2cell = {v: k for k, v in self.cell2id.items()}

        self.PREPROC_FILE = cnf.AUX_PREPROC_FILE % p_CELL_SIZE
        [self.token2cell, self.top_k_map, self.inv_tm, self.train_size] = \
            pickle.load(open(self.PREPROC_FILE, "rb")) if os.path.isfile(self.PREPROC_FILE) else [None, None, None,
                                                                                                  None]
        try:
            with open(cnf.TOPK_THRES_FILE % (cnf.CELL_SIZE, cnf.EPS), 'rb') as pickle_file:
                self.top_k_size = pickle.load(pickle_file)
        except Exception as e:
            print("error in reading top k threshold file: ", e)
            self.top_k_size = 0

        if self.token2cell is not None:
            self.cell2token = {v: k for k, v in self.token2cell.items()}

        nl = utils.get_neighbors(cnf.MAX_HOP_NUM)
        self.nl_size = len(nl)
        self.neighbor_map = dict(zip(nl, range(0, self.nl_size)))

        # for error calculation (this is only an approximation with pythagoras but should be good enough as long as we have a small map)
        self.inv_nm = {v: k for k, v in self.neighbor_map.items()}

        self.neighbor_err = np.zeros((self.nl_size, self.nl_size))
        for c1 in self.inv_nm:
            for c2 in self.inv_nm:
                self.neighbor_err[c1, c2] = np.linalg.norm(
                    self.cell_size * (np.array(self.inv_nm[c1]) - np.array(self.inv_nm[c2])), ord=2)

        self.max_len = None
        self.top_k_err = None
        self.scale_factors = 1. / np.array(
            [self.MAP_HEIGHT, self.MAP_WIDTH, self.MAP_HEIGHT, self.MAP_WIDTH, self.MAX_SLOTS])
        self.projector = utils.Projector(cnf.MIN_X, cnf.MIN_Y, cnf.MAX_X, cnf.MAX_Y)

    def map_to_top_k(self, all_locations, top_locations):
        nearest = {}
        for i in all_locations:
            nearest[i] = min(top_locations, key=lambda x: cell_dist(self.projector, self.id2cell[i], self.id2cell[x]))
        return nearest

    def add_aux_data(self, trace, cnt):
        _get_hour = lambda ts: datetime.fromtimestamp(ts).hour

        dst = trace[-1][0]
        hour = Counter(_get_hour(x[1]) for x in trace).most_common(1)[0][0]
        return [(x[0], dst, hour) for x in trace]

    # thres is used to truncate each trace (to have bounded sensitivity in DP)
    def get_sequence_of_tokens(self, in_corpus, thres, sigma):
        grams = []
        lens = []
        cnt = Counter()
        thres_ind_topk = 0
        count_singles = 0
        # store location frequency for tokenization
        for line in in_corpus:
            if cnf.PREPROC_REMOVE_REPETITIVE:
                line = [line[0]] + [line[i] for i in range(1, len(line)) if line[i][0] != line[i - 1][0]]

        
            l = set(tuple(y) for y in line)
            if len(l) < 2:
                count_singles += 1
                continue
            """
            # Remove repetitive locations
            if cnf.PREPROC_REMOVE_REPETITIVE:
                c_rep = 0
                line_new = [line[0]]
                #line = [line[0]] + [line[i] for i in range(1, len(line)) if line[i][0] != line[i - 1][0]]
                for i in range(1, len(line)):
                        if line[i][0] == line[i - 1][0]:
                            c_rep += 1
                            if c_rep > 10:
                                continue
                            else:
                                line_new += [line[i]]
                        else:
                            c_rep = 0
                            line_new += [line[i]]
            """
            cnt.update(x[0] for x in line)
        print("count singles: ", count_singles)
        # save location frequencies for plotting
        with open(cnf.LOCATION_FREQUENCY_FILE % cnf.TAXI_NUM, "wb") as f:
            pickle.dump(cnt.most_common(), f)
        
        # tokenization
        all_words = add_noise_to_hist(cnt.most_common(), sigma * thres)
        if cnf.PREPROC_MAP_TO_TOP_K:
            #print("topk set after gauss: ",used_words)
            #usedset=set(used_words)
            #print("orig-used: ",topkset-usedset)
            #print("used-orig: ",usedset-topkset)

            #check largest histo bin of locations that contains 80-90% of data:
            histo_thres = cnf.THRESHOLD_PERCENTAGE*sum(cnt.values())
            print("histogram threshold: ", histo_thres)
            sum_of_hists = 0
            for c in range(len(all_words)):
                sum_of_hists += cnt[all_words[c]]
                if sum_of_hists < histo_thres:
                    continue
                else:
                    with open(cnf.TOPK_THRES_FILE % (cnf.CELL_SIZE, cnf.EPS), "wb") as m:
                        pickle.dump(c, m)
                    self.top_k_size = c
                    thres_ind_topk = c
                    print("histogram threshold index at ", cnf.THRESHOLD_PERCENTAGE," is ", c, " sum of thres: ", sum_of_hists, "all counts: ", sum(cnt.values()))
                    break

            used_words = all_words[:thres_ind_topk]
            top_k_locations = [self.id2cell[location] for location in used_words]
            self.top_k_map = dict(zip(top_k_locations, range(thres_ind_topk)))

            # map locations to nearest location in top-K
            nearest_in_top_k = self.map_to_top_k(all_words, used_words)
        else:
            used_words = all_words
        self.cell2token = dict(zip(used_words, range(1, len(used_words) + 1)))
        assert 0 not in self.cell2token, "0 is reserved for padding!"
        # For padding
        self.cell2token[0] = 0
        self.token2cell = {v: k for k, v in self.cell2token.items()}
        print("Number of occurring spatio-temporal points:", len(all_words))

        # mapping to tokens
        labels = []
        dropped = 0
        #dist = Counter()
        init_data = []
        user_id = 0
        trace_lens = []
        traces = []
        dropped_top_k = 0
        repetition = []
        single = 0
        topk_err = 0
        for line in in_corpus:
            # map locations in trace to top K locations, drop traces with distance over threshold
            if cnf.PREPROC_MAP_TO_TOP_K:
                try:
                    top_k_line = [(nearest_in_top_k[x[0]], x[1]) for x in line]
                except:
                    print(line)
                    topk_err += 1
                if any(cell_dist(self.projector, self.id2cell[x[0]], self.id2cell[y[0]]) > cnf.TOP_K_MAPPING_THRESHOLD
                       for x, y in zip(line, top_k_line)):
                    dropped_top_k += 1
                    continue
                line = top_k_line

            # get hour for timestamps
            line = self.add_aux_data(line, cnt)
            
            # Remove repetitive locations
            if cnf.PREPROC_REMOVE_REPETITIVE:
                line = [line[0]] + [line[i] for i in range(1, len(line)) if line[i][0] != line[i - 1][0]]
                """
                c_rep = 0
                line_new = [line[0]]
                for i in range(1, len(line)):
                        if line[i][0] == line[i - 1][0]:
                            c_rep += 1
                            if c_rep > 10:
                                continue
                            else:
                                line_new += [line[i]]
                        else:
                            c_rep = 0
                            line_new += [line[i]]
                line = line_new
                """
            #count repetitive locations
            repetition += list(Counter(line).values())        
            #print("rep: ",repetition) 
            #print(Counter(line).values())
            # we don't keep traces with a single cell
            if len(line) < 2:
                single += 1
                continue

            # Just to write out the original data to file
            traces.append([self.cell2token[line[0][1]], line[0][2], [self.cell2token[x[0]] for x in line]])

            trace_lens.append(len(line))

            # Distribution of starting locations
            #dist[line[0][0]] += 1
            # tokenized_list = [(0,0,0)] * GRAM_SIZE + line
            # Just for test, must be replaced with proper sampling of the very first location (see below)!!!:
            tokenized_list = [(0, 0, 0)] * (cnf.GRAM_SIZE - 1) + line

            # init_data:
            # predictor: time (used_id is just for sampling, it will be removed) label: (src, dst)
            init_data.append([self.cell2token[line[0][0]], self.cell2token[line[0][1]], line[0][2], user_id])

            if thres is not None:
                tokenized_list = tokenized_list[:thres]
            # Remove the last location (test)
            # tokenized_list = tokenized_list[:-1]
            for i in range(0, len(tokenized_list) - cnf.GRAM_SIZE):
                gram = tokenized_list[i:i + cnf.GRAM_SIZE + 1]

                next_loc = gram[-1][0]
                curr_loc = gram[-2][0]
                dst_loc = gram[-1][1]

                # Skip grams if the last location equals the destination
                #if next_loc == dst_loc:
                    #continue

                # Keep the gram only if the next location is within MAX_HOPS
                try:
                    if cnf.PREPROC_MAP_TO_TOP_K:
                        # Next hop w/o localized coding w/o destination:
                        item = self.top_k_map[self.id2cell[next_loc]]
                    else:
                        # Only next hop with localized coding:
                        item = (
                            utils.get_neighbor_code(self.id2cell[curr_loc], self.id2cell[next_loc], self.neighbor_map)
                        )
                    labels.append(item)
                except KeyError as e:
                    dropped += 1
                    continue

                # tokenizing the rest of the gram
                # Zero is kept for padding, we need to predict at least len(top_words) + 1 locations (including padding)
                # elements; next-hop, destination, time (label is different, see above)
                gram = [(self.cell2token[x[0]], self.cell2token[x[1]], x[2]) for x in gram[:-1]]

                # just for sampling; Store the position of every gram in the dataset per user
                gram.append(user_id)
                grams.append(gram)
                lens.append(len(tokenized_list))

            user_id += 1
        
        rep_sort = sorted(repetition)
        re = int(0.97 * len(repetition))
        rep_thres = rep_sort[re]

        print("number of errors with topk projection in preproc.py: ", topk_err)
        print("repetitions stats: ", "mean: ", np.mean(repetition), "median: ", np.median(repetition), "std: ", np.std(repetition),"min: ",min(repetition), "max: ", max(repetition),"mode: ", st.mode(repetition), "at 97% threshold: ", rep_thres)

        print("Dropped grams due to localized coding:", dropped)
        utils.print_stat("Trace length after tokenization", trace_lens)
        print("Dropped 1-long traces:", single)
        if cnf.PREPROC_MAP_TO_TOP_K:
            print("Dropped traces due to top-K mapping:", dropped_top_k)

        # Distribution of starting symbols (first location of the trace)
        # locs, probs = zip(*dist.most_common())
        # probs = np.array(probs) / sum(probs)
        # Draw a random starting location
        # print (np.random.choice(locs, 3, p = probs))

        pickle.dump(traces, open(cnf.ORIG_TRACES_FILE % cnf.CELL_SIZE, "wb"))
        f_in = open(cnf.ORIG_FOR_NGRAM % cnf.CELL_SIZE, "w")
        for _, _, trace in traces:
            f_in.write(" ".join([str(x) for x in trace]) + "\n")
        f_in.close()


        out = open(cnf.ORIG_FOR_ADA % cnf.CELL_SIZE, "w")
        for i, (_, _, trace) in enumerate(traces):
            trace_tmp = [self.id2cell[self.token2cell[x]] for x in trace]
            # lon_idx = int(metric_coordinates[1] / cnf.CELL_SIZE) + 1
            # lat_idx = int(metric_coordinates[0] / cnf.CELL_SIZE) + 1
            # we put the coordinate in the middle of the cell
            mapped_trace = ["%s,%s" % ((x[0] - 0.5) * cnf.CELL_SIZE, (x[1] - 0.5) * cnf.CELL_SIZE) for x in trace_tmp]

            out.write('#%d:\n' % i)
            out.write('>0:' + ";".join(mapped_trace) + ";\n")
        out.close()

        return grams, labels, lens, init_data, thres_ind_topk

    def __scale_data(self, x, inv):
        f = 1. / self.scale_factors if inv else self.scale_factors
        for i in range(x.shape[1]):
            x[:, i] = x[:, i] * f[i]

    def scale_data(self, x):
        self.__scale_data(x, inv=False)

    def unscale_data(self, x):
        self.__scale_data(x, inv=True)

    def plot_init_data(self, name, data):
        utils.plot_init_data(name, data, self.MAP_HEIGHT, self.MAP_WIDTH, self.MAX_SLOTS, self.cell_size)

    def convert_init_data_to_coords(self, data):
        return np.array(
            [[*self.id2cell[self.token2cell[rec[0]]], *self.id2cell[self.token2cell[rec[1]]], rec[2]] for rec in data])

    def load_data(self, sigma, train_size=0.9):
        h5file = open_file(cnf.MAPPED_OUTPUT_DATA_FILE % self.cell_size, mode="r")
        traces = h5file.root.Traces
        total_grams = sum([len(trace) for trace in traces])
        print("Total grams:", total_grams)
        print("Total traces:", len(traces))
        
        #get threshold for sensitivity, where 80% of traces is included get largest length of trace
        traces_sort = sorted(traces, key=len)
        place = int(cnf.THRESHOLD_PERCENTAGE * len(traces))
        #place = int(np.percentile(traces_sort,80))
        trunc_thres = len(traces_sort[place])

        self.id2cell = {v: k for k, v in self.cell2id.items()}

        print("Lenght truncation threshold:", trunc_thres)
        inp_sequences, labels, lens, init_train_data, ind_topk = self.get_sequence_of_tokens(traces, trunc_thres, sigma)
        self.train_size = len(init_train_data)
        print("Number of top K locations:", ind_topk)
        
        if cnf.PREPROC_MAP_TO_TOP_K:
            # for error calculation
            self.inv_tm = {v: k for k, v in self.top_k_map.items()}
            
            self.top_k_err = np.zeros((ind_topk, ind_topk))
            for c1 in self.inv_tm:
                for c2 in self.inv_tm:
                    self.top_k_err[c1, c2] = cell_dist(self.projector, self.inv_tm[c1], self.inv_tm[c2])

        # Dump params for generation and evaluation
        pickle.dump([self.token2cell, self.top_k_map, self.inv_tm, self.train_size], open(self.PREPROC_FILE, "wb"))

        print("Remaining grams:", len(inp_sequences))
        print("MAX_HOP:", cnf.MAX_HOP_NUM)
        self.max_len = max(lens)

        print("Max words num:", len(self.cell2token))
        print("Maximum number of grams per trip:", self.max_len)
        print("Number of grams per trip (stat); mean len: %.3f, median: %.3f, std.dev: %.3f" % (
            np.mean(lens), np.median(lens), np.std(lens)))

        # Creating test/train sets (with shuffling)
        train_data, test_data, train_labels, test_labels = train_test_split(inp_sequences, labels,
                                                                            test_size=1 - train_size, random_state=42)

        # Remove user_id from the end of the grams
        # DO NOT shuffle the data afterwards
        sample_locs_nh = defaultdict(list)
        sample_locs_init = defaultdict(list)

        for i, sample in enumerate(train_data):
            sample_locs_nh[sample.pop()].append(i)

        for i, sample in enumerate(init_train_data):
            sample_locs_init[sample.pop()].append(i)

        test_data = [x[:-1] for x in test_data]

        # Convert dict to list
        sample_locs_nh = list(sample_locs_nh.values())
        sample_locs_init = list(sample_locs_init.values())

        del inp_sequences
        del labels
        train_data = np.float32(train_data)
        init_train_data = np.float32(init_train_data)
        test_data = np.float32(test_data)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        # Dump data for evaluation
        pickle.dump(init_train_data, open(cnf.ORIG_INIT_DATA_FILE % cnf.CELL_SIZE, "wb"))

        # Normalize location into [0,1] otherwise VAE does not always converge with regression (loss: MSE)
        # self.scale_data(init_train_data)

        if len(train_data) > 0:
            print("Train data shape:", train_data.shape, train_data.dtype)
            print("Train label shape:", train_labels.shape, train_labels.dtype)

        if len(test_data) > 0:
            print("Test shape:", test_data.shape, test_data.dtype)
            print("Test label shape:", test_labels.shape, test_labels.dtype)

        h5file.close()

        return train_data, train_labels, test_data, test_labels, init_train_data, sample_locs_nh, sample_locs_init
