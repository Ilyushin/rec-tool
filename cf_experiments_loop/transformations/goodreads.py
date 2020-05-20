import requests
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

CHUNK_SIZE = 32768


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def get_goodreads_data(destination='goodreads.csv'):
    download_file_from_google_drive(id='1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon',
                                    destination=destination)

    data = pd.read_csv(destination)
    print(data)
    os.remove(destination)
    data = pd.DataFrame({'user_id': data['user_id'],
                         'item_id': data['book_id'],
                         'rating': data['rating']})

    train_data, test_data = train_test_split(data, test_size=0.2)
    return train_data, test_data, len(np.unique(data.user_id)), len(np.unique(data.item_id))
