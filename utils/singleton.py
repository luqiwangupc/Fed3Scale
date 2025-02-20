import threading
from typing import Type, TypeVar
from functools import wraps

T = TypeVar('T')


def singleton(cls: Type[T]) -> Type[T]:
    """
    单例装饰器，可以通过装饰器方式将任何类变成单例
    """
    original_new = cls.__new__
    instance = None
    lock = threading.Lock()

    @wraps(cls.__new__)
    def __new__(cls_, *args, **kwargs):
        nonlocal instance
        if instance is None:
            with lock:
                if instance is None:
                    if original_new is object.__new__:
                        instance = original_new(cls_)
                    else:
                        instance = original_new(cls_, *args, **kwargs)
        return instance

    cls.__new__ = staticmethod(__new__)
    return cls
