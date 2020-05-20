import os
import shutil
import urllib
import zipfile
import six
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
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


def _transform_csv(input_path, output_path, names, skip_first, separator=","):
    """Transform csv to a regularized format.

    Args:
      input_path: The path of the raw csv.
      output_path: The path of the cleaned csv.
      names: The csv column names.
      skip_first: Boolean of whether to skip the first line of the raw csv.
      separator: Character used to separate fields in the raw csv.
    """
    if six.PY2:
        names = [six.ensure_text(n, "utf-8") for n in names]

    with tf.io.gfile.GFile(output_path, "wb") as f_out, \
            tf.io.gfile.GFile(input_path, "rb") as f_in:

        # Write column names to the csv.
        f_out.write(",".join(names).encode("utf-8"))
        f_out.write(b"\n")
        for i, line in enumerate(f_in):
            if i == 0 and skip_first:
                continue  # ignore existing labels in the csv

            line = six.ensure_text(line, "utf-8", errors="ignore")
            fields = line.split(separator)
            if separator != ",":
                fields = ['"{}"'.format(field) if "," in field else field
                          for field in fields]
            f_out.write(",".join(fields).encode("utf-8"))


def prepare_data(
        dataset_type=None,
        clear=False,
        movielens_path=None
):
    raitings_file = "ratings.csv"
    movies_file = "movies.csv"
    working_dir = os.path.join(movielens_path, dataset_type)
    raitings_file_path = os.path.join(movielens_path, raitings_file)
    movies_file_path = os.path.join(movielens_path, movies_file)

    if clear:
        shutil.rmtree(raitings_file_path, ignore_errors=True)
        shutil.rmtree(movies_file_path, ignore_errors=True)
        shutil.rmtree(working_dir, ignore_errors=True)

    helpers.create_dir(movielens_path)

    data_url = "http://files.grouplens.org/datasets/movielens/"
    url = "{}{}.zip".format(data_url, dataset_type)

    zip_path = os.path.join(movielens_path, "{}.zip".format(dataset_type))

    if not os.path.exists(zip_path):
        zip_path, _ = urllib.request.urlretrieve(url, zip_path)

    zipfile.ZipFile(zip_path, "r").extractall(movielens_path)

    # os.remove(zip_path)


    if dataset_type == ML_1M:
        _transform_csv(
            input_path=os.path.join(working_dir, "ratings.dat"),
            output_path=os.path.join(movielens_path, raitings_file),
            names=RATING_COLUMNS, skip_first=False, separator="::"
        )

        _transform_csv(
            input_path=os.path.join(working_dir, "movies.dat"),
            output_path=os.path.join(movielens_path, movies_file),
            names=MOVIE_COLUMNS, skip_first=False, separator="::"
        )
    else:
        _transform_csv(
            input_path=os.path.join(working_dir, "ratings.csv"),
            output_path=os.path.join(movielens_path, raitings_file),
            names=RATING_COLUMNS, skip_first=False, separator=","
        )

        _transform_csv(
            input_path=os.path.join(working_dir, "movies.csv"),
            output_path=os.path.join(movielens_path, movies_file),
            names=MOVIE_COLUMNS, skip_first=False, separator=","
        )

    # shutil.copyfile(
    #     os.path.join(working_dir, raitings_file),
    #     os.path.join(movielens_path, raitings_file)
    # )
    # shutil.copyfile(
    #     os.path.join(working_dir, movies_file),
    #     os.path.join(movielens_path, movies_file)
    # )
    tf.io.gfile.rmtree(working_dir)

    dataset = pd.read_csv(os.path.join(movielens_path, raitings_file))
    users_number = len(dataset.user_id.unique())
    items_number = len(dataset.item_id.unique())

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    return train_data, test_data, users_number, items_number


def bpr_movielens(dataset_type=None,
                  clear=False,
                  movielens_path=None):
    train_data, test_data, users_number, items_number = prepare_data(dataset_type=dataset_type,
                                                                     clear=clear,
                                                                     movielens_path=movielens_path)

    train_data = bpr_preprocess_data(users=train_data.user_id,
                                     items=train_data.item_id,
                                     rating=train_data.rating,
                                     rating_threshold=3)

    test_data = bpr_preprocess_data(users=test_data.user_id,
                                    items=test_data.item_id,
                                    rating=test_data.rating,
                                    rating_threshold=3)

    return train_data, test_data, users_number, items_number


# print(prepare_data(dataset_type='ml-1m',
#                    clear=True,
#                    movielens_path='/tmp/cf_experiments_loop/dataset/movielens'))


# preprocessing for vaecf
def vae_movielens(dataset_type=None,
                  clear=False,
                  movielens_path=None):
    train_data, test_data, users_number, items_number = prepare_data(dataset_type=dataset_type,
                                                                     clear=clear,
                                                                     movielens_path=movielens_path)

    return vae_preprocess_data(train_data), vae_preprocess_data(
        test_data), users_number, items_number
