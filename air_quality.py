from csv import reader
import os
import numpy as np
'''
001,北京,BeiJing,39.904210,116.407394,1
004,深圳,ShenZhen,22.543099,114.057868,2
006,天津,TianJin,39.084158,117.200982,1
009,广州,GuangZhou,23.129110,113.264385,2
'''

path = './/data//airquality.csv'

def get_individual_beacon_data(beacon_id):
    individual_beacon_data = []
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for entry in csv_reader:
            for i in range(0, len(entry)):
                if entry[i] == 'NULL':
                    entry[i] = '0'

            if str(beacon_id) == entry[0]:
                try:
                    individual_beacon_data.append([float(x) for x in entry[2:]])
                except:
                    pass
    individual_beacon_data = np.asarray(individual_beacon_data)
    print("beacon {}'s data has shape of {}".format(beacon_id, individual_beacon_data.shape))
    return individual_beacon_data

def get_individual_beacon_id():
    beacons = []

    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for entry in csv_reader:
            try:
                entry_id = entry[0]
                beacons.append(entry_id)
            except:
                pass
    beacons = sorted(set(beacons))
    print('there are {} unique beacons in the dataset'.format(len(beacons)))
    return beacons

def get_chunks(L, n):
    return [L[x: x+n] for x in range(0, len(L), n)]

def break_single_beacon_data(beacon_data):
    chunks = get_chunks(beacon_data, 20)
    chunks = np.asarray(chunks[:len(chunks)-2])

    print(chunks.shape)
    return chunks

def get_city_dataset(city):
    beacons = get_individual_beacon_id()
    city_data = []
    for beacon in beacons:
        if beacon[:3] == city:
            beacon_data = get_individual_beacon_data(beacon)
            chunks = break_single_beacon_data(beacon_data)

            for chunk in chunks:
                city_data.append(np.asarray(chunk))

    city_data = np.asarray(city_data)
    print('city {} has data of shape {}'.format(city, city_data.shape))
    return city_data

beijing = get_city_dataset('001')
np.save('data//pre-processed//beijing.npy', beijing)

shenzhen = get_city_dataset('004')
np.save('data//pre-processed//shenzhen.npy', shenzhen)

tianjin = get_city_dataset('006')
np.save('data//pre-processed//tianjin.npy', tianjin)
#print(np.load('data//pre-processed//tianjin.npy').shape)

guangzhou = get_city_dataset('009')
np.save('data//pre-processed//guangzhou.npy', guangzhou)


import pretty_errors
import os
import numpy as np

# load datasets
beijing = np.load('data//pre-processed//beijing.npy')
shenzhen = np.load('data//pre-processed//shenzhen.npy')
tianjin = np.load('data//pre-processed//tianjin.npy')
guangzhou = np.load('data//pre-processed//guangzhou.npy')

def obtain_x_y(city, city_name):
    xs = []
    ys = []
    for sample in city:
        x = []
        for component in sample[:len(sample)-2]:
            norm = np.linalg.norm(component)
            x.append(norm)

        y = np.linalg.norm(sample[len(sample)-1])

        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    np.save('data//pre-processed//' + city_name + '_finalized_x.npy', xs)
    np.save('data//pre-processed//' + city_name + '_finalized_y.npy', ys)

obtain_x_y(beijing, 'beijing')
obtain_x_y(tianjin, 'tianjin')
obtain_x_y(shenzhen, 'shenzhen')
obtain_x_y(guangzhou, 'guangzhou')

def display_data(city_name):
    xs = np.load('data//pre-processed//'+ city_name + '_finalized_x.npy')
    ys = np.load('data//pre-processed//'+ city_name + '_finalized_y.npy')
    print("{} xs has shape {}".format(city_name, xs.shape))
    print("{} ys has shape {}".format(city_name, ys.shape))


display_data('beijing')
display_data('tianjin')
display_data('shenzhen')
display_data('guangzhou')
