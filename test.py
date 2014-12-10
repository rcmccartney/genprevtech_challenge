__author__ = 'mccar_000'

from multiprocessing.pool import Pool
from profilehooks17.profilehooks import *
#Uses:
#@profile
#@coverage
#@timecall


def testf(at):
    i = 0
    for val in at:
        i += 1

@profile
def main():
    threads = 10
    pool = Pool(1)
    output = pool.map(testf, [tuple([0 for _ in range(5000000)]) for _ in range(threads)])
    print("WERE BACK")


if __name__ == '__main__':
    main()