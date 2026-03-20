import os

from urllib.request import urlretrieve


def download_list(tiles, odir, verbose=False):

    if not os.path.exists(odir):
        raise ValueError(f"odir does not exist {odir}")

    with open(tiles, "r") as f:
        lines = f.readlines()

    for url in lines:
        get_tile(url, odir)


def get_tile(url, odir, verbose=False):

    if not os.path.exists(odir):
        raise ValueError(f"odir does not exist {odir}")

    filename = os.path.join(
        odir, url.split("=")[-1]
    )  # The filename is at the end of the URL

    print(f"Get {filename}")

    if os.path.exists(filename):
        print(f"This file already exists: {filename}")
        return

    path, headers = urlretrieve(url, filename.rstrip("\n"))

    if verbose:
        for name, value in headers.items():
            print(name, value)
    else:
        print(path)
