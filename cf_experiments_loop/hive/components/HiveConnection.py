from pyhive import hive
from TCLIService.ttypes import TOperationState
import logging
from cf_experiments_loop.hive.components.Logger import logger


class HiveConnection:
    def __init__(self, hostname, port=10000):
        self._connection = hive.connect(host=hostname, port=port, auth='NOSASL')
        self._cursor = self._connection.cursor()

    def query(self, query: str, into_dict=True) -> dict:
        self.execute(query)

        results = self._cursor.fetchall()

        if not into_dict:
            return results

        column_descriptions = self._cursor.description
        return self.__into_dict(results, list(zip(*column_descriptions))[0])

    def batch_query(self, query: str, batch_size: int, into_dict=True) -> dict:
        self.execute(query)

        column_descriptions = self._cursor.description

        while True:
            results = self._cursor.fetchmany(batch_size)
            if not results:
                break

            if not into_dict:
                yield results
            else:
                yield self.__into_dict(results, list(zip(*column_descriptions))[0])

    def execute(self, query: str) -> None:
        self.log('Executing query {}'.format(query))

        self._cursor.execute(query)
        status = self._cursor.poll().operationState

        while status in (TOperationState.INITIALIZED_STATE, TOperationState.RUNNING_STATE):
            logs = self._cursor.fetch_logs()

            for message in logs:
                self.log(message)

            # If needed, an asynchronous query can be cancelled at any time with:
            # cursor.cancel()

            status = self._cursor.poll().operationState
            self.log(status)

    def close_connection(self):
        self._cursor.close()
        self._connection.close()

    @staticmethod
    def log(message: str):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(message)

    @staticmethod
    def __into_dict(result_set, column_names) -> dict:
        return dict(zip(column_names, list(zip(*result_set))))

