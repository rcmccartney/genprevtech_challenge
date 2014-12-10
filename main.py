__author__ = 'Rob McCartney'

import sys
import os
import errno
from DataBuffer import *

MIN_DAY = 5440
MIN_REC_DAY = 11284
MAX_REC_DAY = 17644
MAX_DAY = 17858


def read_data(file_name):
    with open(file_name) as f:
        f.readline()  # get rid of first line
        data = f.readlines()
    return data


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def parse(events, start):
    all_atroc = set()
    for line in events:
        line = line.split()
        day = int(line.pop(0)) - start
        region = int(line[3])
        all_atroc.add((region, day))
    return all_atroc


def main():
    if len(sys.argv) != 6:
        print("This program is to be executed as follows: python main.py <input folder> "
              "<output folder> <first training day> <first testing day> <last testing day>")
        sys.exit()

    data_folder = sys.argv[1]
    out_folder = sys.argv[2]
    start_train = int(sys.argv[3])
    start_test = int(sys.argv[4])
    end_test = int(sys.argv[5])
    receiver = Receiver(start_train, start_test, end_test)

    make_sure_path_exists(out_folder)

    if start_train < MIN_DAY:
        print("The value of <first training day> parameter must be " + str(MIN_DAY) + " or above.")
        sys.exit(1)

    if start_train < MIN_REC_DAY:
        print("WARNING: The value of <first training day> parameter"
              " is recommended to be " + str(MIN_REC_DAY) + " or above.")

    if start_test < start_train:
        print("The value of <first testing day> parameter must be greater than "
              "or equal to the value of <first training day> parameter.")
        sys.exit(1)

    if end_test < start_test:
        print("The value of <last testing day> parameter must be greater than or"
              " equal to the value of <first testing day> parameter.")
        sys.exit(1)

    if end_test > MAX_REC_DAY:
        print("WARNING: The value of <last testing day> parameter is"
              " recommended to be " + str(MAX_REC_DAY) + " or below.")

    if end_test > MAX_DAY:
        print("The value of <last testing day> parameter must be " + str(MAX_DAY) + " or below.")
        sys.exit(1)

    # read in the data for regions and atrocities
    events_data = read_data(data_folder + os.sep + "events.txt")
    regions_data = read_data(data_folder + os.sep + "regions.txt")
    copy_lst = events_data[:]
    all_atroc = parse(copy_lst, start_train)
    # pass regions on first training day
    receiver.receive_data(2, start_train, regions_data)
    # set up atrocity data as a list for every possible day
    atroc_buf = [[] for _ in range(MAX_DAY)]
    for line in events_data:
        line = line.split()
        day = int(line.pop(0))
        atroc_buf[day].append(line)

    for cur_day in range(start_train, end_test+1):
        print("Current day = ", cur_day)
        data = read_data(data_folder + os.sep + "data_" + str(cur_day) + ".txt")
        receiver.receive_data(1, cur_day, data)
        receiver.receive_data(0, cur_day, atroc_buf[cur_day])  # atroc could be empty list
        if cur_day >= start_test:
            print("Starting testing")
            results = receiver.predict_atrocities(cur_day, all_atroc)
            f = open(out_folder + os.sep + "res_" + str(cur_day) + ".txt", 'w')
            for i in range(len(results)):
                f.write(str(i) + ": " + str(results[i]) + "\n")
            f.close()

    print("#############################")
    print("Final score", receiver.score)
    print("#############################")


if __name__ == '__main__':
    main()