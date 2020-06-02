from abc import ABC


class AbstractContext(ABC):
    def __init__(self, context: dict):
        self._context = context

    def getContext(self) -> dict:
        return self._context
