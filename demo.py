# -*- coding: utf-8 -*-
# Aydın Göze Polat
# Art with image regularization and Pytorch

import argparse

from modules.regularizers import perona_malik_art


parser = argparse.ArgumentParser(description="Art with image regularization")
parser.add_argument(
    "--image", type=str, nargs="?", help="Input image path", default="images/star.jpg"
)

parser.add_argument(
    "--out-folder",
    type=str,
    nargs="?",
    help="Optional output folder path for the image transformations",
    default=None,
)


if __name__ == "__main__":
    args = parser.parse_args()
    perona_malik_art(args.image, args.out_folder)
