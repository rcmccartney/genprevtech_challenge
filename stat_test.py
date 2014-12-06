
# first run python -m cProfile -o output_file.txt [-s sort_order] myscript.py
import pstats

p = pstats.Stats('output_file.txt')

# sorts by cumulative time in a function, and then prints ten most significant lines
print("CUMULATIVE")
p.sort_stats('cumulative').print_stats(10)

#see what functions were looping a lot, and taking a lot of time
print("TIME")
p.sort_stats('time').print_stats(10)

# statistics for only the class init methods
p.sort_stats('file').print_stats('__init__')

# 50% of init calls
print("Dual sort")
p.sort_stats('time', 'cum').print_stats(.5, 'init')

#callers
print("Callers")
p.print_callers(.5, 'init')