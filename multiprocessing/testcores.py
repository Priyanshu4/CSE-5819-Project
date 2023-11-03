import threading, time

def f(finall, l):
    t = sum(l)
    finall.append(t)

d = [[1*i, 2*i, 3*i, 4*i] for i in range(2400000)]


import multiprocessing

def f(finall, l):
    t = sum(l)
    finall.append(t)

if __name__ == "__main__":

    final2 = []
    start = time.time()
    for l in d:
        f(final2, l)
    #print(final2)
    print(time.time()-start)

    final = []
    threads = []

    start = time.time()
    for l in d:
        thread = threading.Thread(target=f, args=(final, l))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print(time.time()-start)

    #print(final)

    
    #print(final2)



