import os
import dill


def save_dict(data, path):
    with open(path, 'wb') as f:
        dill.dump(data, f)


def read_dict(path):
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


