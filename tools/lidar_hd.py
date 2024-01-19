import os

from urllib.request import urlretrieve


def get_tile(url, odir, verbose=False):

    filename = os.path.join(odir, os.path.split(url)[-1])

    print(f'Get {filename}')

    if os.path.exists(filename):
        print(f'This file already exists: {filename}')
        return

    path, headers = urlretrieve(url, filename)

    if verbose:
        for name, value in headers.items():
            print(name, value)
    else:
        print(path)
