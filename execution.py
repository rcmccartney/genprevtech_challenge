__author__ = 'Rob McCartney'

from rand_forest.EntropyFn import *
from rand_forest.forest import *
import sys


def train_dt(datafile, numtrees, depth_limit, learner):

    forest = Forest(depth_limit, filename=datafile, weak_learner=learner)
    forest.add_tree(1, snapshot=True)
    forest.add_tree(numtrees//2, snapshot=True)
    forest.add_tree(numtrees, snapshot=True)
    forest.learning_curve()
    forest.test()
    forest.region_plot()
    return forest


def find_best_settings(trainfile):
    """
    Run this once to get the best k, depth, iterations for your data
    Beware, it takes a long time
    :param trainfile:
    :return:
    """
    minerror = float('NaN')
    x1 = list(range(1, 10, 1))  # go by 1
    x2 = list(range(10, 50, 5))  # go by 5
    x3 = list(range(50, 200, 10))  # go by 10
    for depthlimit in [1, 2, 3]:
        for k in x1+x2+x3:
            for maxiterations in range(150, 700, 50):
                print("Trying depth:", depthlimit, "k:", k, "Iterations:", maxiterations)
                forest = train_dt(maxiterations, depthlimit, trainfile, EntropyFn(k, minval, maxval))
                print("Error:", forest.error[-1])
                if math.isnan(minerror) or forest.error[-1] < minerror:
                    minerror = forest.error[-1]
                    best_it = maxiterations
                    best_k = k
                    best_depth = depthlimit
    print("######\n#Best results - Depth:", best_depth, "k:", best_k, "Iterations:", best_it, "\n######")
    return best_it, best_k, best_depth


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: python execution.py <trainfile> <trees> <depth> <splits> <minsplit> <maxsplit> [<testfile>]")
        sys.exit()

    #uncomment this to find best settings
    #best_it, best_k, best_depth = find_best_settings(sys.argv[1])

    my_forest = train_dt(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                         EntropyFn(k=int(sys.argv[4]), minsplit=int(sys.argv[5]), maxsplit=int(sys.argv[6])))

    if len(sys.argv) > 7:
        my_forest.test(sys.argv[7])
        my_forest.region_plot(testfile=sys.argv[7])