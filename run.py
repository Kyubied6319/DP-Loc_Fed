import os
import sys
import time
import utils


cnf = utils.load_cfg("cfg/cfg_general.json")


def run_sim(cfg, cell_size):
    print(cfg, cell_size)
    # Step 2: Train for your life...
    if os.system("runVAE.sh VAE %d %s" % (cell_size, cfg)):
        sys.exit(1)
    if os.system("runTRACE.sh TRACES %d %s" % (cell_size, cfg)):
        sys.exit
    #Step 3: Generate results for your paper...
    if os.system("python generator.py VAE %d %s > gen.txt" % (cell_size, cfg)):
        sys.exit(1)
    
    if os.system("python generator.py TRACES %d %s" % (cell_size, cfg)):
        sys.exit(1)
    # Step 4: Evaluate your performance...
    for m in [0, 1, 5, 10, 25, 50, 100, 150]: 
    	if os.system("python evaluate.py %d %s %d" % (cell_size, cfg, m)):
            sys.exit(1)
    

# Cell size conversions: 315 -> 250 m, 632 -> 500 m, 1264
for cell_size in [632]:

    print("======================== ", cnf.INPUT_DIR, "=================================")
    print("======================== CELL SIZE: ", cell_size, "=================================")
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("STARTING TIME: ", time_string)

    # Step 1: Map taxi GPS data to a grid (THESE BELOW HAVE BEEN UNCOMMENTED)
    if os.system("python create_mapped_data.py %f" % cell_size):
        sys.exit(1)
    
    #print("=================================================================== EPS = 0.5 ================================================================")
    #run_sim("cfg/cfg_eps05.json", cell_size=cell_size)
    
    #print("=================================================================== EPS = 1 ==================================================================")
    #run_sim("cfg/cfg_eps1.json", cell_size=cell_size)

    #print("=================================================================== EPS = 2 ==================================================================")
    run_sim("cfg/cfg_eps2.json", cell_size=cell_size)
    
    #print("=================================================================== EPS = 5 ==================================================================")
    #run_sim("cfg/cfg_eps5.json", cell_size=cell_size)
    
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("FINISH TIME: ", time_string)

# LA FIN
