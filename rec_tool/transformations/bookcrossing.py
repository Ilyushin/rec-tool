"""
Bookcrossing dataset transformation methods
"""
import os
import shutil
import urllib
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def download_bookcrossing(url='http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip',
                          zip_path='bookcrossing.zip'):
    """
    :param url:
    :param zip_path:
    :return:
    """
    zip_path, _ = urllib.request.urlretrieve(url, zip_path)
    zipfile.ZipFile(zip_path, "r").extractall('bookcrossing')
    os.remove(zip_path)


def bookcrossing_converting():
    """
    :return: users, items, ratings
    """

    download_bookcrossing()
    train_data = pd.read_csv('bookcrossing/BX-Book-Ratings.csv', delimiter=';', encoding='latin1')

    curusers = list(set(train_data["User-ID"]))
    users_uuid_int_dict = dict(zip(curusers, range(len(curusers))))

    curitems = list(set(train_data["ISBN"]))
    items_uuid_int_dict = dict(zip(curitems, range(len(curitems))))

    train_data["User-ID"] = train_data["User-ID"].apply(lambda x: users_uuid_int_dict[x])
    train_data["ISBN"] = train_data["ISBN"].apply(lambda x: items_uuid_int_dict[x])
    train_data["Book-Rating"] = train_data["Book-Rating"].apply(lambda x: int(x))

    shutil.rmtree('bookcrossing')

    data = pd.DataFrame({'user_id': train_data["User-ID"],
                         'item_id': train_data["ISBN"],
                         'rating': train_data["Book-Rating"]})

    train, test = train_test_split(data, test_size=0.2)
    return train, test, len(np.unique(data.user_id)), len(np.unique(data.item_id))
