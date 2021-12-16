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

def get_city_dataset(city):
    beacons = get_individual_beacon_id()
    city_data = []
    for beacon in beacons:
        if beacon[:3] == city:
            beacon_data = get_individual_beacon_data(beacon)
            city_data.append(beacon_data)

    city_data = np.asarray(city_data)
    print('city {} has data of shape {}'.format(city, city_data.shape))
    return city_data

get_city_dataset('001')
