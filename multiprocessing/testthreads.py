import threading
import os, time


# Get the number of CPU cores (logical processors)
num_threads = os.cpu_count()

def f(x):
    return x*x

def thread_task(start, end, result_list):
    for i in range(start, end):
        result_list.append(f(i))

if __name__ == '__main__':
    result = []
    total_range = 240000000

    # Using threads
    start = time.time()
    threads = []

    for i in range(num_threads):
        thread_start = i * (total_range // num_threads)
        thread_end = (i + 1) * (total_range // num_threads)

        thread = threading.Thread(target=thread_task, args=(thread_start, thread_end, result))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Time taken with threads:", time.time() - start)

    # Using the built-in map function for comparison
    start = time.time()
    result = list(map(f, range(total_range)))
    print("Time taken without threads:", time.time() - start)
