from multiprocessing import Pool
import os, time


# Get the number of CPU cores (logical processors)
num_cores = os.cpu_count()

def f(x):
    return sum(x)

if __name__ == '__main__':
    l = [[x*1, x*2, x*3, x*4] for x in range(24)]
    print(list(zip(,range(5))))
    print(l)
    with Pool(num_cores) as p:
        start = time.time()
        result = p.map(f, l)
        print(time.time() - start)
        print(result)
    
    start = time.time()
    result = list(map(f, l))
    print(time.time() - start)