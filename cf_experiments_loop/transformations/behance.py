import pandas as pd
import numpy as np


def behance_converting(file_path='Behance_appreciate_1M'):

    data = np.asarray(pd.read_csv(file_path, header=None))
    users, items, rating = [], [], []
    for row in data:
        row_data = row[0].split(' ')
        users.append(int(row_data[0]))
        items.append(int(row_data[1]))
        rating.append(1)

    return pd.DataFrame({'user_id': users, 'item_id': items, 'rating': rating})
