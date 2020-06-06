import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def behance_converting():

    data = np.asarray(pd.read_csv('Behance_appreciate_1M', header=None))
    users, items, rating = [], [], []
    for row in data:
        row_data = row[0].split(' ')
        users.append(int(row_data[0]))
        items.append(int(row_data[1]))
        rating.append(1)

    data = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': rating})
    train_data, test_data = train_test_split(data, test_size=0.2)

    return train_data, test_data, len(np.unique(data.user_id)), len(np.unique(data.item_id))
