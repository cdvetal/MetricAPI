import cv2
import numpy as np
from torchvision import transforms


def load_image(data):
    # convert string of image data to uint8
    nparr = np.frombuffer(data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensorImg = transforms.ToTensor()(img).unsqueeze(0)

    return tensorImg
