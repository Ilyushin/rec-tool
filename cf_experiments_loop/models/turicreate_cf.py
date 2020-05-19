import turicreate as tc


def train_turicreate(train_data_set, test_data_set, num_users, num_items):

    train_set = tc.SFrame(train_data_set)

    popularity_model = tc.popularity_recommender.create(
        train_set,
        user_id='user_id',
        item_id='movie_id',
        target='rating'
    )
    ranking_model = tc.ranking_factorization_recommender.create(
        train_data_set,
        target='views_count',
        solver='ials'
    )

    recommendations_pop = popularity_model.recommend(users=test_data_set.users, k=10)
    recommendations_rank = ranking_model.recommend(users=test_data_set.users, k=10)

    return recommendations_pop, recommendations_rank

