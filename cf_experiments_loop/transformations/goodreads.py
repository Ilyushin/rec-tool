"""
Goodreads dataset transformation methods
"""
import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


CHUNK_SIZE = 32768


def download_file_from_google_drive(doc_id, destination):
    """
    :param doc_id: str
    :param destination: str: path for .csv file
    :return:
    """
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': doc_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': doc_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    """
    :param response:
    :return:
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    """
    :param response:
    :param destination: str: path for .csv file
    :return:
    """

    with open(destination, "wb") as input_file:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:   # filter out keep-alive new chunks
                input_file.write(chunk)


def get_goodreads_data(destination='goodreads.csv'):
    """
    :param destination: str: path for .csv file
    :return:
    """
    download_file_from_google_drive(doc_id='1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon',
                                    destination=destination)

    data = pd.read_csv(destination)
    os.remove(destination)
    data = pd.DataFrame({'user_id': data['user_id'],
                         'item_id': data['book_id'],
                         'rating': data['rating']})

    train_data, test_data = train_test_split(data, test_size=0.2)
    return train_data, test_data, len(np.unique(data.user_id)), len(np.unique(data.item_id))
