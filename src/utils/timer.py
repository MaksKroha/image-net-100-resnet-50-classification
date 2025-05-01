import time

def timed(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        print(f"- {function.__name__} {end_time - start_time} s")
        return result
    return wrapper