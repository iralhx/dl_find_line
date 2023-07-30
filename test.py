from userloss import *
import torch
import torch.nn as nn

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
# Predict with the model
results = model('bus1.jpg')  # predict on an image


a=1