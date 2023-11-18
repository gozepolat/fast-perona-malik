# -*- coding: utf-8 -*-
# Aydın Göze Polat, 2017
# Image regularizers as PyTorch modules

import os
import torch
import torch.nn.functional as F
from torch import nn, optim, squeeze
from torch.autograd import Variable

from modules.utils import (
    scalar_to_parameter,
    make_operator,
    image_to_variable,
    normalize,
    imshow,
    imsave,
)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier(m.bias.data)


def perona_malik_art(image_path, out_folder=None):
    print("Perona-Malik model...")
    image = image_to_variable(image_path)
    _, image_name = os.path.split(image_path)
    image_base, ext = image_name.split(".")

    pm = PeronaMalik(
        image.size(),
        diffusion_rate=0.2,
        delta_t=0.2,
        coefficient="exp",
        learn_operator=True,
    )

    lr = 0.000002
    optimizer = optim.SGD(
        pm.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0005
    )
    if torch.cuda.is_available():
        pm.cuda()

    original = normalize(image)

    out = pm.forward(original)
    k = 0
    for i in range(100001):
        optimizer.zero_grad()
        out = Variable(out.data, requires_grad=False)
        out = pm.forward(out)
        loss = torch.sum(
            torch.pow((out - original), 2) + torch.abs(pm.gradients) + torch.pow(out, 2)
        )
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 75 == 0:
            print(f"Iteration {i}")
            transformed = squeeze(squeeze(out, 0), 0).cpu().data.numpy()

            imshow(transformed)
            if out_folder:
                imsave(os.path.join(out_folder, f"{image_base}_{k}.{ext}"), transformed)
                k += 1

            # update the learning rate and refresh the operator
            if i % 375 == 300:
                pm.laplace = make_operator("laplace", requires_grad=True)
                lr *= 0.995
                optimizer = optim.SGD(
                    pm.parameters(),
                    lr=lr,
                    momentum=0.9,
                    dampening=0,
                    weight_decay=0.0005,
                )
                out = out * 0.6 + original * 0.4


def perona_malik_vanilla(image):
    print("Perona-Malik model...")
    image = image_to_variable(image)

    pm = PeronaMalik(image.size(), diffusion_rate=0.2, delta_t=0.01, coefficient="exp")
    if torch.cuda.is_available():
        pm.cuda()
    original = normalize(image)
    out = pm.forward(original)

    for i in range(5001):
        pm.zero_grad()
        out = pm.forward(Variable(out.data))
        if i % 100 == 0:
            print("iteration %d" % i)
            imshow(squeeze(squeeze(out, 0), 0).cpu().data.numpy())


class PeronaMalik(nn.Module):
    def get_exp_diffusion_coefficient(self):
        return torch.exp(-torch.pow(self.gradients, 2) / self.diffusion_rate_2)

    def get_inv_diffusion_coefficient(self):
        return torch.pow(self.gradient_magnitude / self.diffusion_rate_2 + 1, -1)

    def pick_coefficient(self, coefficient):
        # pick the method which will calculate the diffusion speed
        if coefficient == "exp":
            self.coefficient = self.get_exp_diffusion_coefficient
        elif coefficient == "inv":
            self.coefficient = self.get_inv_diffusion_coefficient

    def get_conv_diffusion(self, image, padded_image):
        # experimental, doesn't work well
        return image * F.tanh(F.conv2d(1 - padded_image, self.conv1_params))

    def init_weights(self, layer):
        nn.init.orthogonal(layer.weight)
        nn.init.constant(layer.bias, 0.0)

    def __init__(
        self, size, diffusion_rate, delta_t, coefficient="exp", learn_operator=False
    ):
        super(PeronaMalik, self).__init__()
        self.gradients = None
        self.delta_t = delta_t
        self.gradient_magnitude = None

        # allow learning parameters using requires_grad=True
        self.diffusion_rate_2 = scalar_to_parameter(
            diffusion_rate**2, size, requires_grad=False
        )
        self.conv1_params = make_operator("laplace", requires_grad=True)

        self.laplace = make_operator("laplace", requires_grad=learn_operator)
        self.scharr_x = make_operator("scharr_x")
        self.scharr_y = make_operator("scharr_y")

        self.coefficient = None
        self.pick_coefficient(coefficient)

    def forward(self, image):
        # ReflectionPad2d for the image boundaries instead of padding=1
        padded_image = nn.ReflectionPad2d(1)(image)
        self.gradients = F.conv2d(padded_image, self.laplace)

        if self.coefficient == self.get_inv_diffusion_coefficient:
            grad_x = F.conv2d(padded_image, self.scharr_x)
            grad_y = F.conv2d(padded_image, self.scharr_y)
            self.gradient_magnitude = torch.sqrt(
                torch.pow(grad_x, 2) + torch.pow(grad_y, 2)
            )

        if self.coefficient is None:
            diffusion = self.gradients * torch.pow(
                torch.abs(self.get_conv_diffusion(image, padded_image))
                / self.diffusion_rate_2
                + 1,
                -1,
            )
        else:
            diffusion = self.gradients * self.coefficient()
        image = image + diffusion * self.delta_t

        return torch.clamp(image, min=0, max=1.0)
