data = pickle.load(open(rep_init_file, "rb"))

data_vis = preproc.convert_init_data_to_coords(data)

preproc.plot_init_data("vae_DP", data_vis)

