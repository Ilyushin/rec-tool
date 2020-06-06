from cf_experiments_loop.hive.provider.context import AbstractContext


class LocalDataContext(AbstractContext):
    def getPath(self):
        return self._context.get('path')
