import pickle
import sys
from datetime import datetime
from math import ceil, log2
from time import time

from pandas.tseries.holiday import USFederalHolidayCalendar
from tables import *

from utils import *

cnf = load_cfg("cfg/cfg_general.json")
cnf.CELL_SIZE = int(float(sys.argv[1]))


# for aggregation by time (returns consecutive slots with identical timestamps)
def get_next_time_slot(trace):
    locs = []
    prev_time = trace[0][1]
    for (loc, ts) in trace:
        if prev_time != ts:
            yield locs, prev_time
            locs = []
            prev_time = ts
        locs.append(loc)
    yield locs, prev_time


def is_weekend_holiday(holidays, timestamp):
    date_time = datetime.fromtimestamp(int(timestamp))
    return (date_time.weekday() in (5, 6)) or (date_time in holidays)


# compute grid position
def generate_grid_pos(longitude, latitude, projector):
    metric_coordinates = projector.toProjected(longitude, latitude)
    # grid (0,0) is not used (0 values are reserved for padding)
    lon_idx = int(metric_coordinates[1] / cnf.CELL_SIZE) + 1
    lat_idx = int(metric_coordinates[0] / cnf.CELL_SIZE) + 1

    # cell_id = lon_idx * MAP_WIDTH + lat_idx
    cell_id = (lon_idx, lat_idx)
    return cell_id


# For Szilvia
def get_flagged_points_trim_gps(holidays, visits):
    traces = []
    prev_flag = 0
    trace_temp = []
    for i, visit in enumerate(visits):
        lat, lon = float(visit[0]), float(visit[1])
        flag, timestamp = int(visit[2]), int(visit[3])
        if cnf.MIN_X <= lon <= cnf.MAX_X and cnf.MIN_Y <= lat <= cnf.MAX_Y:
        #if not is_weekend_holiday(holidays, timestamp) \
                #and cnf.MIN_X <= lon <= cnf.MAX_X and cnf.MIN_Y <= lat <= cnf.MAX_Y:
            if flag == 1:
                prev_flag = flag
                trace_temp.append(((lon, lat), timestamp))
            elif (flag == 0 and prev_flag == 1) or (flag == 1 and i == len(raw_trace) - 1):
                prev_flag = 0
                if cnf.GRAM_SIZE < len(trace_temp):
                    traces.append(trace_temp)
                trace_temp = []
    return traces


# filter by speed
def is_plausible_trace(trace, speed_thres):
    for i in range(len(trace) - 1):
        s = haversine(trace[i][0], trace[i + 1][0])
        dt = trace[i + 1][1] - trace[i][1]
        try:
            speed = (3600 * s / dt)
        except:
            continue
        if speed >= speed_thres:
            return False

    return True


'''
def get_speeds(trace):
    ret = []
    for i in range(len(trace) - 1):
        s = haversine(trace[i][0], trace[i + 1][0])
        dt = trace[i + 1][1] - trace[i][1]
        ret.append(3600 * s / dt)
    return ret
'''

# Simple quantization
quantize = lambda val, q: val - val % q


# ------ Main -------

def main():
    # Mercator projection to map GPS into a plane
    projector = Projector(cnf.MIN_X, cnf.MIN_Y, cnf.MAX_X, cnf.MAX_Y)

    # height: 94, width: 58
    MAP_HEIGHT = int(ceil(projector.toProjected(cnf.MAX_X, cnf.MAX_Y)[1] / cnf.CELL_SIZE))
    MAP_WIDTH = int(ceil(projector.toProjected(cnf.MAX_X, cnf.MAX_Y)[0] / cnf.CELL_SIZE))
    MAX_SLOTS = 24
    print("MAP HEIGHT: ", MAP_HEIGHT)
    print("MAP WIDTH: ", MAP_WIDTH)

    cells = [(x, y) for x in range(MAP_HEIGHT + 1) for y in range(MAP_WIDTH + 1)]

    cell2id = dict(zip(cells, range(1, len(cells) + 1)))

    # Order of the Hilbert curve
    HILB_ORDER = int(ceil(log2(max(MAP_WIDTH, MAP_HEIGHT))))

    # Cell size in meter (a cell is a rectangle and not a square)
    cell_size = 1000 * max(haversine(projector.toGPS((cnf.CELL_SIZE, 0)), [cnf.MIN_X, cnf.MIN_Y]),
                           haversine(projector.toGPS((0, cnf.CELL_SIZE)), [cnf.MIN_X, cnf.MIN_Y]))
    print("Cell side 1: %.3f m " % (1000 * haversine(projector.toGPS((cnf.CELL_SIZE, 0)), [cnf.MIN_X, cnf.MIN_Y])))
    print("Cell side 2: %.3f m" % (1000 * haversine(projector.toGPS((0, cnf.CELL_SIZE)), [cnf.MIN_X, cnf.MIN_Y])))
    print("Cell size used for error calculation:", cell_size)

    '''
    pbar = tqdm(total=len(cells)**2)
    distances = np.zeros((len(cells), len(cells)), dtype=np.int16)
    for i, c1 in enumerate(cells):
        for j, c2 in enumerate(cells):
            #print ("Distance (%d, %d) - (%d, %d): %.3f" % (c1[0], c1[1], c2[0], c2[1], cell_dist(projector, c1, c2)))
            distances[i,j] = cell_dist(projector, c1, c2)
            pbar.update(1)
    
    pbar.close()
    pickle.dump([cells, cell2id, distances], open("cell_distances.pickle", "wb"))
    '''
    pickle.dump([MAP_WIDTH, MAP_HEIGHT, MAX_SLOTS, cell2id, cell_size],
                open(cnf.AUX_CELL_INFO_FILE % cnf.CELL_SIZE, "wb"))

    ALL_CABS = sort_files(cnf.INPUT_DIR, desc=True, fnames_only=True)
    #ALL_CABS = open(r"C:\Users\Edward\Documents\PProj\DP-Loc\datasets\Porto\Porto_preprocessed.txt", "r")

    t_start = time()

    holidays = USFederalHolidayCalendar().holidays(start='2008-01-01', end='2008-12-31').to_pydatetime()

    # For Output
    fileh = open_file(cnf.MAPPED_OUTPUT_DATA_FILE % cell_size, mode="w")
    out_traces = fileh.create_vlarray(fileh.root, 'Traces', UInt32Atom(shape=(cnf.DIM_SIZE,)), "Traces")
    out_traces.flavor = 'python'

    # For trace size stats
    trace_lens_before_agg = []
    trace_lens_after_agg = []
    cab_num = len(ALL_CABS)
    dropped = 0

    speeds = []
    # Process every taxi file
    kept = []
    for cab_id, CAB_FILE in enumerate(ALL_CABS[:1]):
        print('Cab %d/%d' % (cab_id, cab_num))

        with open(CAB_FILE, 'r') as source:
            # this gives a list
            raw_records = source.readlines()

            # cab_name = CAB_FILE[len(INPUT_DIR):]

        # input: 37.79674 -122.42616 1 1211019247
        cab_visits = [record.rstrip().split(' ') for record in raw_records]
        # sort by timestamp
        #cab_visits.sort(key=lambda record: record[3], reverse=False)
        # split to taxi rides
        cab_traces = get_flagged_points_trim_gps(holidays, cab_visits)
        orig = len(cab_traces)
        # filter by speed
        cab_traces = [trace for trace in cab_traces if is_plausible_trace(trace, cnf.MAX_SPEED)]

        '''
        for trace in cab_traces:
            speeds.extend(get_speeds(trace))
    
        continue
        '''

        trace_lens_before_agg.extend([len(x) for x in cab_traces])

        #kept.append(len(cab_traces))
        #print("Dropped %d out of %d trips (%.2f%%). Non-cont: %d. Kept: %d. Total trips so far: %d" % (
            #orig - len(cab_traces), orig,
            #100 * (1 - len(cab_traces) / orig) if orig > 0 else 0, dropped, len(cab_traces),
            #len(trace_lens_before_agg)))

        # aggregation by time and location transformation
        traces_tmp = []
        for trace in cab_traces:
            # quantize time stamps & map to grid
            trace = [(generate_grid_pos(visit[0][0], visit[0][1], projector), quantize(visit[1], cnf.AGG_TIME)) for
                     visit in trace]
            new_trace = []
            for (locs, ts) in get_next_time_slot(trace):
                if len(locs) > 0:
                    new_trace.append((Counter(locs).most_common(1)[0][0], ts))

            # Interpolation
            i = 1
            while i < len(new_trace):
                last_pos, next_pos = np.asarray(new_trace[i - 1][0]).astype(float), np.asarray(new_trace[i][0]).astype(
                    float)
                start_ts, end_ts = new_trace[i - 1][1], new_trace[i][1]
                #print("start", start_ts, " final ", end_ts)
                diff = int(abs(start_ts - end_ts) / cnf.AGG_TIME)
                #print(diff)
                gap_len = diff - 1
                if 1 < diff <= cnf.MAX_GAP_LEN:
                    diff_vec = (next_pos - last_pos) / (gap_len + 1)
                    # linear interpolation
                    new_trace[i:i] = [(tuple(np.asarray(last_pos + (j + 1) * diff_vec).astype(int)),
                                       start_ts + (j + 1) * cnf.AGG_TIME)
                                      for j in range(gap_len)]
                    i += gap_len
                elif diff > cnf.MAX_GAP_LEN:
                    dropped += 1
                    break
                i += 1

            if diff > cnf.MAX_GAP_LEN:
                continue

            # map to David Hilbert
            # new_trace = [(point_to_hilbert_curve(visit[0][0], visit[0][1], HILB_ORDER) + 1, visit[1]) for visit in new_trace]
            # for stats
            #print("line 231", len(new_trace))
            trace_lens_after_agg.append(len(new_trace))
            # new_trace = [(visit[0][0], visit[0][1]) for visit in new_trace]
            new_trace = [(cell2id[visit[0]], visit[1]) for visit in new_trace]
            # new_trace = [(visit[0][0], visit[0][1], visit[1]) for visit in new_trace]

            # Output to h5 file
            out_traces.append(new_trace)

    fileh.close()

    # print (speeds)
    # pickle.dump(speeds, open("speeds.pickle", "wb"))

    # print ("Dropped lengths:", np.mean(diffs), max(diffs), min(diffs), np.median(diffs), np.std(diffs))
    print("Non-contiguous trips (dropped):", dropped)
    
    print_stat("Summary before aggregation (trace_len)", trace_lens_before_agg)
    print("trace len after", len(trace_lens_after_agg))
    print_stat("Summary after aggregation (trace_len)", trace_lens_after_agg)

    #print_stat("Summary of kept traces (num_of_traces)", kept)

    t_end = time()
    print("Full running time of preprocessing: {}s".format(round(t_end - t_start), 3))


if __name__ == '__main__':
    main()
