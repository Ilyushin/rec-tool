import pickle
from datetime import datetime, timedelta
from .builder.Builder import Builder
from .provider.DateRangeDataProvider import DateRangeDataProvider
from .provider.context.DateRangeContext import DateRangeContext
from .components.HiveConnection import HiveConnection
from .components.Logger import logger


period = 14
model_path = "./data/model"
data_path = "./data/data"
api_data_path = "./data/api_data"
hive_host = "localhost"
hive_port = 10000
es_host = ""


def buildModel(end_date, period, hive_host, hive_port):
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
    return oDataProvider


buildModel(end_date=datetime.strptime('01-03-2020', '%d-%m-%Y'), period=period, hive_host=hive_host, hive_port=hive_port)
