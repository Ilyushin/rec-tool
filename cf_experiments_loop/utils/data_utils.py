"""Classes and operations related to processing data.
"""

import numpy as np
from io import BytesIO
from io import StringIO
from zipfile import ZipFile
import requests


def get_zip_file(url, filepath):
    """Gets zip file from url.

    Args:
        url: A string, the url of zip file.
        filepath: A string, the file path inside the zip file.

    Returns:
        A String, the content of wanted file.
    """

    zipfile = ZipFile(BytesIO(requests.get(url).content))
    file = zipfile.open(filepath).read().decode('utf8')

    return StringIO(file)


class BatchGenerator(object):
    """Generator for data.
    """

    def __init__(self, features, targets=None, batch_size=1024, shuffle=True):
        if targets is not None and features.shape[0] != targets.shape[0]:
            raise ValueError('The shape 0 of x should '
                             'be equal to that of y. ')

        self.x = features
        self.y = targets
        self.length = targets.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def next(self):
        """
        :return:
        """
        start = end = 0
        length = self.length
        batch_size = self.batch_size

        if self.shuffle:
            permutation = np.random.permutation(length)
            self.x = self.x[permutation]
            self.y = self.y[permutation]

        flag = False
        while not flag:
            end += batch_size

            if end > length:
                end = length - 1
                flag = True

            yield self._get_batch(start, end)

            start = end

    def _get_batch(self, start, end):
        """
        :param start:
        :param end:
        :return:
        """
        if self.y is not None:
            return self.x[start:end], self.y[start:end]
        else:
            return self.x[start:end]
