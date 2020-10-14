"""
Turicreate Collaborative Filtering model
"""
import turicreate as tc
import pandas as pd
from sklearn.model_selection import train_test_split


def train_turicreate(data_path=''):
    """
    :return:
    """

    data = tc.SFrame(data_path)

    data_df = pd.DataFrame({'user_id': data['user_id'],
                            'item_id': data['item_id'],
                            'rating': data['views_count']})

    curusers = list(set(data_df["user_id"]))
    users_uuid_int_dict = dict(zip(curusers, range(len(curusers))))

    curitems = list(set(data_df["item_id"]))
    items_uuid_int_dict = dict(zip(curitems, range(len(curitems))))

    data_df["user_id"] = data_df["user_id"].apply(lambda x: users_uuid_int_dict[x])
    data_df["item_id"] = data_df["item_id"].apply(lambda x: items_uuid_int_dict[x])
    data_df["rating"] = data_df["rating"].apply(lambda x: int(x))

    train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)
    train_data, test_data = tc.SFrame(train_data), tc.SFrame(test_data)

    popularity_model = tc.popularity_recommender.create(
        tc.SFrame(train_data),
        user_id='user_id',
        item_id='movie_id',
        target='rating'
    )

    ranking_model = tc.ranking_factorization_recommender.create(
        train_data,
        target='views_count',
        solver='ials'
    )

    pop_evaluation = popularity_model.evaluate_rmse(dataset=test_data, target='rating')
    rank_evaluation = ranking_model.evaluate_rmse(dataset=test_data, target='rating')

    print('pop_evaluation', pop_evaluation)
    print('rank_evaluation', rank_evaluation)


