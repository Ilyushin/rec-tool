import turicreate as tc
import os
from cf_experiments_loop.hive.provider import AbstractDataProvider
from cf_experiments_loop.hive.provider.context import AbstractContext
from cf_experiments_loop.hive.provider.context import LocalDataContext
from cf_experiments_loop.hive.components.Logger import logger


class LocalDataProvider(AbstractDataProvider):
    def __init__(self):
        pass

    def supportsContext(self, context: AbstractContext) -> bool:
        return isinstance(context, LocalDataContext)

    def _doGetData(self, context: LocalDataContext) -> tc.SFrame:
        logger.debug('Loading data from path {}'.format(context.getPath()))

        if os.path.exists(context.getPath()) is False:
            raise RuntimeError('Path {} for loading data does not exist'.format(context.getPath()))

        return tc.load_sframe(context.getPath())
