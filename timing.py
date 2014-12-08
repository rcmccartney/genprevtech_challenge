import atexit
import functools
from time import clock

line = "="*40


def seconds2str(t):
    return "%d:%02d:%02d.%03d" % \
        functools.reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t*1000,), 1000, 60, 60])


def log(s, elapsed=None):
    print(line)
    print(seconds2str(clock()), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line, end="\n\n")


def endlog():
    end = clock()
    elapsed = end-start
    log("End Program", seconds2str(elapsed))


def now():
    return seconds2str(clock())

start = clock()
atexit.register(endlog)
log("Start Program")