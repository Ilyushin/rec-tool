"""
Movielens dataset transformation methods
"""

import os
import shutil
import urllib
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from signal_transformation import helpers
from cf_experiments_loop.models.bpr_model import bpr_preprocess_data
from cf_experiments_loop.models.vae import vae_preprocess_data

ML_1M = "ml-1m"
ML_20M = "ml-20m"

GENRE_COLUMN = "genres"
ITEM_COLUMN = "item_id"  # movies
RATING_COLUMN = "rating"
TIMESTAMP_COLUMN = "timestamp"
TITLE_COLUMN = "titles"
USER_COLUMN = "user_id"

RATING_COLUMNS = [USER_COLUMN, ITEM_COLUMN, RATING_COLUMN, TIMESTAMP_COLUMN]
MOVIE_COLUMNS = [ITEM_COLUMN, TITLE_COLUMN, GENRE_COLUMN]


def prepare_data(dataset_type=None,
                 movielens_path=None):

    """
    :param dataset_type:
    :param movielens_path:
    :return:
    """

    if dataset_type == 'ml-1m':
        ratings_file = "ratings.dat"
    else:
        ratings_file = 'ratings.csv'

    working_dir = os.path.join(movielens_path, dataset_type)
    ratings_file_path = os.path.join(working_dir, ratings_file)

    helpers.create_dir(movielens_path)

    data_url = "http://files.grouplens.org/datasets/movielens/"
    url = "{}{}.zip".format(data_url, dataset_type)

    zip_path = os.path.join(movielens_path, "{}.zip".format(dataset_type))

    if not os.path.exists(zip_path):
        zip_path, _ = urllib.request.urlretrieve(url, zip_path)

    zipfile.ZipFile(zip_path, "r").extractall(movielens_path)

    os.remove(zip_path)

    if dataset_type == 'ml-1m':
        dataset = pd.read_csv(ratings_file_path, delimiter='::')
    else:
        dataset = pd.read_csv(ratings_file_path)
    print(dataset)
    dataset.columns = RATING_COLUMNS

    curusers = list(set(dataset["user_id"]))
    users_uuid_int_dict = dict(zip(curusers, range(len(curusers))))

    curitems = list(set(dataset["item_id"]))
    items_uuid_int_dict = dict(zip(curitems, range(len(curitems))))

    dataset["user_id"] = dataset["user_id"].apply(lambda x: users_uuid_int_dict[x])
    dataset["item_id"] = dataset["item_id"].apply(lambda x: items_uuid_int_dict[x])
    dataset["rating"] = dataset["rating"].apply(lambda x: int(x))

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    shutil.rmtree(movielens_path, ignore_errors=True)
    return train_data, test_data, len(dataset.user_id.unique()), len(dataset.item_id.unique())


def bpr_movielens(dataset_type=None,
                  movielens_path=None):
    """
    :param dataset_type:
    :param movielens_path:
    :return:
    """
    train_data, test_data, users_number, items_number = prepare_data(dataset_type=dataset_type,
                                                                     movielens_path=movielens_path)

    train_data = bpr_preprocess_data(users=train_data.user_id,
                                     items=train_data.item_id,
                                     rating=train_data.rating)

    test_data = bpr_preprocess_data(users=test_data.user_id,
                                    items=test_data.item_id,
                                    rating=test_data.rating)

    return train_data, test_data, users_number, items_number


# preprocessing for vaecf
def vae_movielens(dataset_type=None,
                  movielens_path=None):
    """
    :param dataset_type:
    :param movielens_path:
    :return:
    """
    train_data, test_data, users_number, items_number = prepare_data(dataset_type=dataset_type,
                                                                     movielens_path=movielens_path)

    return vae_preprocess_data(train_data), vae_preprocess_data(
        test_data), users_number, items_number
