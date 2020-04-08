import importlib


def fn(name):
    module, method = name.rsplit('.', 1)
    return getattr(importlib.import_module(module), method)
