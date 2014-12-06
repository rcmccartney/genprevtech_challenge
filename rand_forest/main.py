__author__ = 'mccar_000'
import sys 
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
    receiver = Receiver()

    events_data = read_data(data_folder + "/events.txt")
    regions_data = read_data(data_folder + "/regions.txt")

    atroc_buf = []
    for line in events_data:
        line = line.split()
        day = int(line.pop(0))
        if day in atroc_buf:
            atroc_buf[day].append(line)
        else:
            atroc_buf[day] = [line]

    for i in range(len(atroc_buf)):
        receiver.receive_data(2, start_train, regions_data)

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

    for cur_day in range(start_train, end_test+1):
        print("Day = ", cur_day)
        data = read_data(data_folder + "/data_" + str(cur_day) + ".txt")
        receiver.receive_data(1, cur_day, data)
        receiver.receive_data(0, cur_day, atroc_buf[cur_day])

        if cur_day >= start_test:
            results = receiver.predict_atrocities(cur_day)
            f = open(out_folder + "/res_" + str(cur_day) + ".txt", 'w')
            for result in results:
                f.write(result + "\n")
            f.close()

if __name__ == '__main__':
    main()