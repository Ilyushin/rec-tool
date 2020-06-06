import turicreate as tc
from abc import ABCMeta, abstractmethod
from cf_experiments_loop.hive.provider.context import AbstractContext


class AbstractDataProvider(object, metaclass=ABCMeta):
    def getData(self, context: AbstractContext) -> tc.SFrame:
        if self.supportsContext(context=context) is False:
            raise RuntimeError('Context is not supported')
        return self._doGetData(context=context)

    @abstractmethod
    def supportsContext(self, context: AbstractContext) -> bool:
        return True

    @abstractmethod
    def _doGetData(self, context: AbstractContext) -> tc.SFrame:
        pass
