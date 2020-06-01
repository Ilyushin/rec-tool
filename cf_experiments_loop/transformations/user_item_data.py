import turicreate as tc
from sklearn.model_selection import train_test_split
from cf_experiments_loop.hive.download_hive_data import buildModel


def get_most_active_users(data_set, number_users):
    """Get most active users.

        Args:

        Returns:

        Raises:

        """
    distinct_article_sf = data_set.groupby(['user_id'],
                                           operations={
                                               'articles_count': tc.aggregate.COUNT('item_id')
                                           })
    views_count_sf = data_set.groupby(['user_id'],
                                      operations={
                                          'number_views': tc.aggregate.SUM('views_count')
                                      })
    aggregate_data_set = distinct_article_sf.join(views_count_sf, on='user_id', how='inner')
    return (aggregate_data_set.sort([('articles_count', False), ('number_views', False)])
            .head(number_users))['user_id'].unique()


def user_item_transform(end_date, period, data_path, users_number):

    # download data
    buildModel(end_date=end_date, period=period, data_path=data_path)

    # data uploading
    data = tc.load_sframe(data_path).to_dataframe()

    # data preprocessing
    data = get_most_active_users(data_set=data, number_users=users_number)
    data.columns = ['user_id', 'item_id', 'rating']

    users_number = len(data.user_id.unique())
    items_number = len(data.item_id.unique())

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data, users_number, items_number