import timeit


def runtime(f):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = f(*args, **kwargs)
        end = timeit.default_timer()
        print('Elapsed Time : %.6fms' % ((end - start) * 1000))
        return result
    return wrapper
