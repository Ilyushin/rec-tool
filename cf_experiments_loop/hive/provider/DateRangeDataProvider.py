import turicreate as tc
from cf_experiments_loop.hive.provider import AbstractDataProvider
from cf_experiments_loop.hive.provider.context import DateRangeContext, AbstractContext
from cf_experiments_loop.hive.components import HiveConnection
from cf_experiments_loop.hive.components.Logger import logger


class DateRangeDataProvider(AbstractDataProvider):
    def __init__(self, hive_connection: HiveConnection):
        self._hive_connection = hive_connection

    def supportsContext(self, context: AbstractContext) -> bool:
        return isinstance(context, DateRangeContext)

    def _doGetData(self, context: DateRangeContext) -> tc.SFrame:
        o_hive_connection = self._hive_connection

        base_query = """
                SELECT
                    `from_article_id`,
                    `user_id`,
                    `hour`,
                    `date`
                FROM
                    `{}`
                WHERE
                    `date` >= "{}"
                    AND `date` <= "{}"
                    AND `is_in_gdpr_area` = 0
                    AND `user_id` NOT LIKE "b0057e12%"
                    AND `user_id` NOT LIKE "xxxxxx%"
            """.format('{}', context.getStartDate().strftime('%Y-%m-%d'), context.getEndDate().strftime('%Y-%m-%d'))

        sub_query = base_query.format('sponsored_impressions')

        if context.includeSponsoredData():
            sub_query = '{} UNION ALL {}'.format(sub_query, base_query.format('organic_impressions'))

        # Count only once a pair that occurred in a date/hour combination
        query = """
                SELECT
                    `from_article_id` as `item_id`,
                    `user_id`,
                    COUNT(DISTINCT (`hour`, `date`)) as `views_count`
                FROM ({}) AS q1
                GROUP BY
                    `from_article_id`,
                    `user_id`
            """.format(sub_query)

        tmp_table_name = 'recommendation_service_aggregated_tmp'
        tmp_table_query = 'CREATE TEMPORARY TABLE `{}` AS {}'.format(tmp_table_name, query)

        logger.info('Creating temporary table {}'.format(tmp_table_name))

        o_hive_connection.execute(query=tmp_table_query)
        logger.info('Temporary table {} created'.format(tmp_table_name))

        logger.info('Calculating percentile for `views_count`')
        percentile_query = """
                SELECT 
                    PERCENTILE(`views_count`, 0.999) AS `calculated_percentile` 
                FROM {}
            """.format(tmp_table_name)
        calculated_percentile_result = o_hive_connection.query(query=percentile_query)
        calculated_percentile = calculated_percentile_result.get('calculated_percentile')[0]

        logger.info('Percentile for `views_count` = {}'.format(calculated_percentile))

        tmp_table_name2 = 'recommendation_service_aggregated_tmp2'

        number_of_articles_query = """
                CREATE TEMPORARY TABLE {} AS 
                SELECT
                    `t`.`user_id`,
                    `t`.`item_id`,
                    `t`.`views_count`,
                    `s`.`number_of_articles`
                FROM {} `t`
                JOIN (
                    SELECT 
                        `user_id`, 
                        COUNT(*) as `number_of_articles`
                    FROM {}
                    WHERE `views_count` < {}
                    GROUP BY `user_id`
                    HAVING `number_of_articles` > 1
                ) as `s`
                ON (`t`.`user_id` = `s`.`user_id`)
                WHERE 
                    `t`.`views_count` < {}
            """.format(tmp_table_name2, tmp_table_name, tmp_table_name, calculated_percentile, calculated_percentile)

        logger.info('Obtaining number of unique articles for each user')

        logger.info('Creating temporary table {}'.format(tmp_table_name2))
        o_hive_connection.execute(query=number_of_articles_query)
        logger.info('Temporary table {} created'.format(tmp_table_name2))

        logger.info('Dropping temporary table {}'.format(tmp_table_name))
        o_hive_connection.execute(query='DROP TABLE {}'.format(tmp_table_name))
        logger.info('Dropped temporary table {}'.format(tmp_table_name))

        logger.info('Calculating percentile for `number_of_articles`')

        number_of_articles_percentile_query = """
                SELECT 
                    PERCENTILE(`number_of_articles`, 0.8512) as `calculated_percentile` 
                FROM {}
            """.format(tmp_table_name2)
        number_of_articles_percentile_result = o_hive_connection.query(query=number_of_articles_percentile_query)
        number_of_articles_percentile = number_of_articles_percentile_result.get('calculated_percentile')[0]

        logger.info('Percentile for `number_of_articles` = {}'.format(number_of_articles_percentile))

        rows_query = """
                SELECT 
                    `user_id`, 
                    `item_id`, 
                    `views_count` 
                FROM {} 
                WHERE 
                    `number_of_articles` < {} 
                    AND `views_count` < {}
            """.format(tmp_table_name2, number_of_articles_percentile, calculated_percentile)

        logger.info('Obtaining results...')

        batch_size = 100000
        train_data_set = tc.SFrame()

        for results in o_hive_connection.batch_query(query=rows_query, batch_size=batch_size, into_dict=False):
            local_sFrame = tc.SFrame(data=results)
            train_data_set = train_data_set.append(other=local_sFrame)
            logger.info('Appended {} rows to sFrame'.format(batch_size))

        logger.info('Results obtained')

        logger.info('Dropping temporary table {}'.format(tmp_table_name2))
        o_hive_connection.execute(query='DROP TABLE {}'.format(tmp_table_name2))
        logger.info('Dropped temporary table {}'.format(tmp_table_name2))
        o_hive_connection.close_connection()

        logger.info('Unpacking data frame')

        train_data_set = train_data_set.unpack('X1')
        train_data_set = train_data_set.rename({'X1.0': 'user_id', 'X1.1': 'item_id', 'X1.2': 'views_count'})

        return train_data_set
