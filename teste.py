import cv2
import numpy as np
import torch
from torchvision import transforms

from metrics import SaturationFitness


fitness = SaturationFitness()

img = cv2.imread('flowers.jpeg')

tensorImg = transforms.ToTensor()(img).unsqueeze(0)


print(fitness.evaluate(tensorImg))
