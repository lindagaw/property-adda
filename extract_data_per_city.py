from csv import reader
import numpy as np
import csv
import sys

def equal_sized_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

# read city csv
def read_city_csv(city_csv):
    city_csv_as_list = []
    with open(city_csv, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        for row in csv_reader:
            city_csv_as_list.append(row)

    chunks_raw = equal_sized_chunks(city_csv_as_list, 20)

    chunks = []
    for chunk in chunks_raw:
        temp = ''
        flag = False
        triggered = False
        for single in chunk:
            if not flag:
                temp = single[8]
                flag = True
            else:
                if not temp == single[8]:
                    triggered = True

            for item in single:
                if item == 'NULL':
                    triggered = True

        if not triggered:
            chunks.append(chunk)

    return chunks[:len(chunks)-2]

city = sys.argv[1]

print('processing data from the city {} ...'.format(city))

city_csv = 'data//' + city + '_combine_meteorology_and_airquality.csv'
chunks = read_city_csv(city_csv)


# pm2.5 index 10, weather index 2, temperature index 3, pressure index 4
# humidity index 5, windspeed index 6, winddirection index 7
def extract_features(chunks, index, name):

    print('extracting the features of {} ...'.format(name))
    overall_xs = []
    overall_ys = []
    for chunk in chunks:
        xs = []
        ys = []
        # chunk is of shape (20, 16)
        for i in range(0, len(chunk)):
                if i < len(chunk)-1:
                    xs.append(float(chunk[i][index]))
                else:
                    ys.append(float(chunk[i][index]))
        overall_xs.append(xs)
        overall_ys.append(ys)

    overall_xs = np.asarray(overall_xs).squeeze()
    overall_ys = np.asarray(overall_ys).squeeze()

    print('xs has shape {}'.format(overall_xs.shape))
    print('ys has shape {}'.format(overall_ys.shape))

    np.save('data//pre-processed//' + city + '_' + name + '_xs.npy', overall_xs)
    np.save('data//pre-processed//' + city + '_' + name + '_ys.npy', overall_ys)

    return overall_xs, overall_ys

extract_features(chunks, index=10, name='pm25')
extract_features(chunks, index=2, name='weather')
#extract_features(chunks, index=3, name='temperature')
#extract_features(chunks, index=4, name='pressure')
#extract_features(chunks, index=5, name='humidity')
extract_features(chunks, index=6, name='windspeed')
#extract_features(chunks, index=7, name='winddirection')
