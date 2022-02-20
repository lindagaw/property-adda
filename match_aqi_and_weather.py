from datetime import datetime, timedelta
from csv import reader
import csv

meteorology_csv = 'data//meteorology.csv'
air_quality_csv = 'data//airquality.csv'

def str_to_datetime(date_time_str):
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    return date_time_obj

def match_time():
    appropriate_list = []
    with open(meteorology_csv, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        for row in csv_reader:
            if int(row[0]) == 1 or int(row[0]) == 4 or int(row[0]) == 6 or int(row[0]) == 9:
                appropriate_list.append(row)

    return appropriate_list

def combine_meteorology_and_airquality():
    appropriate_list = match_time()
    appropriate_list_v2 = []
    with open(air_quality_csv, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)

        for row in csv_reader:
            print(row[:3])
            if row[:3] == '001' or row[:3] == '004' or row[:3] == '006' or row[:3] == '009':
                for item in appropriate_list:
                    if int(item[0]) == int(row[0][2]) and \
                        str_to_datetime(item[1]) - str_to_datetime(row[1]) < timedelta(hours=2) and \
                        str_to_datetime(item[1]) - str_to_datetime(row[1]) > timedelta(hours=0):
                            appropriate_list_v2.append(item + row)
                            print(item+row)
                            break


    return appropriate_list_v2

combine_meteorology_and_airquality_list = combine_meteorology_and_airquality()
head_airquality = ['Station ID', 'Time', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
head_meterology = ['ID', 'Time', 'Weather', 'Temperature', 'Pressure', 'Humidity', 'Wind Speed', 'Wind Direction']
heads = head_airquality + head_meterology

with open('data//combine_meteorology_and_airquality.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(heads)
    write.writerows(combine_meteorology_and_airquality_list)
