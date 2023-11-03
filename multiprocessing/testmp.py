from multiprocessing import Pool
import os, time


# Get the number of CPU cores (logical processors)
num_cores = os.cpu_count()

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(num_cores) as p:
        start = time.time()
        result = p.map(f, range(240000000))
        print(time.time() - start)
    
    start = time.time()
    result = list(map(f, range(240000000)))
    print(time.time() - start)