
def torch_transform(users, items, rating):
    """ Transform data for torch models
    :param users: list
    :param items: list
    :param rating: list
    :return:
    """
    return [(users[i], items[i], rating[i]) for i in range(len(rating))]