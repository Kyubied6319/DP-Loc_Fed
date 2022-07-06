import numpy as np
from collections import Counter
import pickle

# ngram
for c in [315, 632]:
    for eps in [1.00, 2.00]:
        [trip_size_jsd, fs_result, density] = pickle.load(open("out_ngram1/results-cell_%d-eps_%.2f.pickle" % (c, eps), "rb"))
        print ("--> Cell size: %d Eps: %.2f" % (c, eps))
        print ("JSD:", trip_size_jsd)
        print ("density:", density)
        for (k,v) in fs_result.items():
            print ("TOP-%d: %.2f" %(k,v))

xxx
for c in [315, 632]:
    for eps in [1.00, 2.00]:
        [trip_size_jsd_per_hour, fs_results, density_per_hour, trip_dist_per_hour] = pickle.load(open("out/results-cell_%d-eps_%.2f.pickle" % (c, eps), "rb"))
        print ("--> Cell size: %d Eps: %.2f" % (c, eps))
        for (k,v) in fs_results['overall'].items():
            print ("TOP-%d: %.2f" %(k,v))


for c in [315, 632]:
    for eps in [1.00, 2.00]:
        print ("--> Cell size: %d Eps: %.2f" % (c, eps))
        [train_loss_results, train_accuracy_results, train_error_results] = pickle.load(open("out/nh_train_metrics-cell_%d-eps_%.2f.pickle" % (c, eps), "rb")) 
        print ("NH Loss:", train_loss_results[-1].numpy())
        print ("NH Accuracy:", train_accuracy_results[-1].numpy())
        print ("NH Error:", train_error_results[-1].numpy())
        
        train_loss_results = pickle.load(open("out/vae_train_metrics-cell_%d-eps_%.2f.pickle" % (c, eps), "rb")) 
        
        print ("VAE Loss:", train_loss_results[-1].numpy())


# For stat printing
def print_stat(name, v):
    print ("---> %s <---" % name)
    print ("Total:", len(v)) 
    print ("mean: %.2f, median: %d, max: %d, min: %d, std.dev: %.2f" % (np.mean(v), np.median(v), max(v), min(v), np.std(v)))

cnt = Counter()
lens =  []
for s in open("orig_for_ngram-500.dat"):
    t = [int(x) for x in s.strip().split()]
    lens.append(len(t))
    cnt.update(t)

print_stat("ngram",lens)
print (len(cnt))