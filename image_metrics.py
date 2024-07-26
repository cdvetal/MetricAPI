import clip
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from torchvision.utils import save_image

from metrics import *
from utils import load_image

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']

app = Flask('image_metrics')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_model = "ViT-B/32"
model_32, preprocess_32 = clip.load("ViT-B/32", device=device)
model_14, preprocess_14 = clip.load("ViT-L/14", device=device)

metrics = {"symmetry": SymmetryFitness(),
           "aesthetic": AestheticFitness(model_32, preprocess_32),
           "aesthetic2": Aesthetic2Fitness(model_14, preprocess_14),
           # "style": StyleLoss(),
           "edge": EdgeFitness(),
           "gaussian": GaussianFitness(),
           # "palette": PaletteFitness(),
           "resmem": ResmemFitness(),
           "smoothness": SmoothnessFitness(),
           "saturation": SaturationFitness(),
           }

clip_prompt = ClipPrompt(model_32, preprocess_32)


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/test', methods=['POST'])
def receive_image():
    tensorImg = load_image(request.data)

    save_image(tensorImg, 'dogcat2.jpeg')

    # do some fancy processing here....

    # build a response dict to send back to client
    response = jsonify({'message': 'image received. size={}x{}'.format(tensorImg.shape[2], tensorImg.shape[3])})

    return response


@app.route('/images/', methods=['POST'])
def image_metrics():
    tensorImg = load_image(request.data)

    with torch.no_grad():
        response_message = {}
        for metric_name, metric in metrics.items():
            metric_value = metric.evaluate(tensorImg)
            metric_value = metric_value.cpu().numpy().squeeze()
            response_message[metric_name] = float(metric_value)

    # build a response dict to send back to client
    response = jsonify(response_message)

    return response


@app.route('/images/<metric>', methods=['POST'])
def image_metric(metric):
    if metric in metrics:
        tensorImg = load_image(request.data)

        with torch.no_grad():
            metric_value = metrics[metric].evaluate(tensorImg)
            metric_value = metric_value.cpu().numpy().squeeze()

            response_message = {metric: float(metric_value)}
    else:
        response_message = {"error": 'metric not found'}

    # build a response dict to send back to client
    response = jsonify(response_message)

    return response


@app.route('/images/clip', methods=['POST'])
def image_metric_clip():
    tensorImg = load_image(request.data)

    prompt = request.args.get('prompt')

    print("Prompt", prompt)

    with torch.no_grad():
        metric_value = clip_prompt.evaluate(tensorImg, prompt)
        metric_value = metric_value.cpu().numpy().squeeze()

        response_message = {"clip": float(metric_value)}

    # build a response dict to send back to client
    response = jsonify(response_message)

    return response

