import time
from functools import wraps

def track_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️ {func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper
