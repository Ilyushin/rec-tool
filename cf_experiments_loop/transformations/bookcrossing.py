import pandas as pd
import pickle
import urllib
import zipfile
import os
import shutil
import numpy as np


def download_bookcrossing(url='http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip',
                          zip_path='bookcrossing.zip'):
    zip_path, _ = urllib.request.urlretrieve(url, zip_path)
    zipfile.ZipFile(zip_path, "r").extractall('bookcrossing')
    os.remove(zip_path)


def bookcrossing_converting(file_path='bookcrossing/BX-Book-Ratings.csv'):
    """
    :param file_path:
    :return: users, items, ratings
    """
    train_data = pd.read_csv(file_path, delimiter=';', encoding='latin1')

    curusers = list(set(train_data["User-ID"]))
    users_uuid_int_dict = dict(zip(curusers, range(len(curusers))))
    curitems = list(set(train_data["ISBN"]))
    items_uuid_int_dict = dict(zip(curitems, range(len(curitems))))

    train_data["User-ID"] = train_data["User-ID"].apply(lambda x: users_uuid_int_dict[x])
    train_data["ISBN"] = train_data["ISBN"].apply(lambda x: items_uuid_int_dict[x])
    train_data["Book-Rating"] = train_data["Book-Rating"].apply(lambda x: int(x))

    shutil.rmtree('bookcrossing')
    print(train_data)

    return train_data["User-ID"], train_data["ISBN"], train_data["Book-Rating"]

