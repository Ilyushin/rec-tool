"""
"""
import importlib


def fn(name):
    """
    :param name:
    :return:
    """
    module, method = name.rsplit('.', 1)
    return getattr(importlib.import_module(module), method)
