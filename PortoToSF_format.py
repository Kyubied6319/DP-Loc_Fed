# converts the raw Porto dataset to the same format as in the San Francisco dataset, ie one line contains one gps cordinate with timestamp and flag

import re

min_lat, max_lat = 90, -90
min_lon, max_lon = 180, -180
file = open('datasets/Porto/raw.csv', 'r')
new_file = open('datasets/Porto/Porto_preprocessed.txt', 'w+')
file.readline()
c = 0
# set how many lines you need with the parameter c, SF taxi has approx 120.000 after preprocessing
while c < 200000:  
    line = file.readline()
    #for line in file:
    split_line = line.split(',')
    c += 1
    ts = int(split_line[5][1:-1])
    trace = re.findall(r"(?:\[)([.\d, -]{1,50})(?:\])", line)
    for i in range(len(trace)):
        gps = trace[i].split(",")
        row = str(gps[1]) + " " + str(gps[0]) + " " + str(1) + " " + str(ts + i*15) + "\n"  # lat, lon is needed for create_mapped_data.py
        new_file.write(row)
        longitude = float(gps[0])
        latitude = float(gps[1])
        """if min_lat > latitude:  # finds the extremes of the GPS coordinates in the dataset, use if you dont know them, but be careful with outliers or anomalies in the data!
            min_lat = latitude
        if max_lat < latitude:
            max_lat = latitude
        if min_lon > longitude:
            min_lon = longitude
        if max_lon < longitude:
            max_lon = longitude"""
    new_file.write(str(latitude) + " " + str(longitude) + " " + str(0) + " " + str(ts + (i+1)*15) + "\n")

print(c)
print("(min_lat;max_lat)=(%f;%f)" % (min_lat, max_lat))
print("(min_lon;max_lon)=(%f;%f)" % (min_lon, max_lon))
file.close()
new_file.close()

