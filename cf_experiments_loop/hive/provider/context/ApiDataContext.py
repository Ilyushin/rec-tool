import turicreate as tc
from cf_experiments_loop.hive.provider.context import AbstractContext


class ApiDataContext(AbstractContext):
    def getSourceData(self) -> tc.SFrame:
        return self._context.get('sourceData')
