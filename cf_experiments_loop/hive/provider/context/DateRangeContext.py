import datetime
from cf_experiments_loop.hive.provider.context.AbstractContext import AbstractContext


class DateRangeContext(AbstractContext):
    def getStartDate(self) -> datetime:
        return self._context.get('startDate')

    def getEndDate(self) -> datetime:
        return self._context.get('endDate')

    def includeSponsoredData(self) -> bool:
        return self._context.get('includeSponsoredData', True)
