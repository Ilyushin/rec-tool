import pickle
from datetime import datetime, timedelta
from .builder.Builder import Builder
from .provider.DateRangeDataProvider import DateRangeDataProvider
from .provider.context.DateRangeContext import DateRangeContext
from .components.HiveConnection import HiveConnection
from .components.Logger import logger


def buildModel(end_date: str,
               period: int,
               data_path: str,
               hive_host="localhost",
               hive_port=10000,
               model_path="./data/model"):

    end_date = datetime.strptime(end_date, '%d-%m-%Y')
    startDate = end_date - timedelta(days=period)

    logger.info('Selected dates between {} and {}'.format(startDate, end_date))

    oHiveConnection = HiveConnection(hostname=hive_host, port=hive_port)
    oContext        = DateRangeContext(context=dict({'startDate': startDate, 'endDate': end_date}))

    oDataProvider   = DateRangeDataProvider(hive_connection=oHiveConnection)
    oData = oDataProvider.getData(oContext)
    oBuilder = Builder(target='views_count')
    oModel = oBuilder.build(oData)

    oModel.save(model_path)
    oData.save(data_path)
    return


buildModel(end_date='01-03-2020', period=14, data_path='./data')
