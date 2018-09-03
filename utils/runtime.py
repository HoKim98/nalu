import timeit


# Source from https://juliahwang.kr/algorism/python/2017/09/12/
# %ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%BD%94%EB%93%9C%EC%8B%A4%ED%96
# %89%EC%8B%9C%EA%B0%84%EC%B8%A1%EC%A0%95%ED%95%98%EA%B8%B0.html
def runtime(f):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = f(*args, **kwargs)
        end = timeit.default_timer()
        print('Elapsed Time : %.6fms' % ((end - start) * 1000))
        return result
    return wrapper
