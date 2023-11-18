# -*- coding: utf-8 -*-
# Aydın Göze Polat, 2017
# Art with image regularization and PyTorch

import torch
from torch.autograd import Variable
from torch import squeeze
from torch import optim
import cv2
import numpy as np

from modules.utils import image_to_variable, normalize
from modules.regularizers import PeronaMalik


def perona_malik_art(video_path=None):
    print("Perona-Malik on drugs...")

    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("camera capture device is not open..")
        exit(-1)

    res, image = cap.read()
    out = image_to_variable(image)
    out = normalize(out)
    pm = PeronaMalik(
        out.size(),
        diffusion_rate=0.5,
        delta_t=0.2,
        coefficient="exp",
        learn_operator=True,
    )
    if torch.cuda.is_available():
        pm.cuda()
    else:
        print(
            "Cuda is not available, it is recommended to utilize GPUs for video processing"
        )

    lr = 0.000000125
    optimizer = optim.SGD(
        pm.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0005
    )
    i = 0

    while True:
        res, image = cap.read()

        if not res:
            print("problem reading the image..")
            exit(-1)

        image = image_to_variable(image)
        image = normalize(image)

        out = pm.forward(out)
        out = Variable(out.data, requires_grad=False)

        if i % 8 != 0:
            out = out * 0.8 + image * 0.2
        else:
            i = 0
        optimizer.zero_grad()
        out = pm.forward(out)

        # make fidelity component less prominent in the loss
        loss = torch.sum(
            torch.pow(pm.gradients * (out - image), 2)
            + pm.gradients
            + 10 * torch.pow(out * (pm.gradients), 2)
        )

        loss.backward(retain_graph=True)
        optimizer.step()
        i += 1
        if i % 2 != 0:
            continue

        cv2.imshow("demo", np.rollaxis(squeeze(out, 0).cpu().data.numpy(), 0, 3))
        cv2.waitKey(1)


if __name__ == "__main__":
    perona_malik_art()
