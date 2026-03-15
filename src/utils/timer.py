import time
from functools import wraps


class Timer:
    """Класс для замера времени выполнения"""
    
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
    def get_elapsed_formatted(self):
        """Возвращает отформатированное время (часы:минуты:секунды)"""
        hours = int(self.elapsed // 3600)
        minutes = int((self.elapsed % 3600) // 60)
        seconds = int(self.elapsed % 60)
        return f"{hours} ч {minutes} мин {seconds} сек"
    
    def print_elapsed(self):
        """Печатает затраченное время"""
        print(f"⏱️ {self.name} заняло: {self.get_elapsed_formatted()}")


def timeit(func):
    """Декоратор для замера времени выполнения функции"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer(func.__name__)
        with timer:
            result = func(*args, **kwargs)
        timer.print_elapsed()
        return result
    return wrapper