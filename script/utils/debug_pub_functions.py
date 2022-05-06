import cProfile
import time
from functools import wraps
from line_profiler import LineProfiler


def func_time(f):
    """
    简单记录执行时间
    :param f:
    :return:
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, 'took', end - start, 'seconds')
        return result

    return wrapper


def func_cprofile(f):
    """
    内建分析器
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = f(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='time')

    return wrapper


try:
    def func_line_time(follow=[]):
        """
        每行代码执行时间详细报告
        :param follow: 内部调用方法
        :return:
        """
        def decorate(func):
            @wraps(func)
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return decorate
except ImportError:
    def func_line_time(follow=[]):
        "Helpful if you accidentally leave in production!"
        def decorate(func):
            @wraps(func)
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return decorate
