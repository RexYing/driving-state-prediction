import csv
import numpy as np
import os
import os.path as path

sensor_fname= '/home/rex/workspace/torcs-data/sensor/108462755.txt'
image_dir = '/home/rex/workspace/torcs-data/image'

with open(sensor_fname, 'rb') as csvfile:
    sensor_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    sensor_data = [row for row in sensor_reader]

image_files = [long(path.splitext(f)[0]) for f in os.listdir(image_dir) if path.isfile(path.join(image_dir, f))]
image_files.sort()
print(image_files)

idx = 0;
synced_sensor_data = []
for image_t in image_files:
    while (np.absolute(long(sensor_data[idx][0]) - image_t) > np.absolute(long(sensor_data[idx+1][0]) - image_t)):
        idx += 1;
    print(idx)
    synced_sensor_data.append(sensor_data[idx])


synced_sensor_fname = sensor_fname[:-4] + '_synced' + sensor_fname[-4:]
with open(synced_sensor_fname, 'wb') as csvfile:
    sensor_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in synced_sensor_data:
        sensor_writer.writerow(row)

