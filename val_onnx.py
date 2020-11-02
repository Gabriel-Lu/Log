'''
Use validation dataset of imagenet to evaluate onnx model.
acc1, acc5, inference time included.


'''

import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np # we're going to use numpy to process input and output data
import onnxruntime  # to inference ONNX models, we use ONNX runtime
import onnx
from onnx import numpy_helper
import urllib.request
import json
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score
import glob

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def validate(val_loader, session):
    correct1 = 0
    correct5 = 0
    total = 0
    total_time = 0.0
    for i, (images, target) in enumerate(val_loader):
        ground_truth = target[0].numpy()    # ground truth label
        # run onnx model to predict label of the given images
        image_data = np.array(images)
        input_data = image_data.reshape(1,3,224,224).astype('float32')
        start = time.time()
        output = session.run([], {input_name: input_data})  # (list)    inference
        end = time.time()
        total_time += end - start
        softmax_output = softmax(np.array(output)).tolist() # numpy
        top5 = np.argpartition(softmax_output, -5)[-5:] # top5 possible 
        top1 = np.argmax(softmax_output)
        # print(top5)
        # print(top1)
        # print(ground_truth)
        # print("___________________________________")
        if top1 == ground_truth:
            correct5 += 1
            correct1 += 1
        elif ground_truth in top5:
            correct5 += 1
        total += 1
        if i != 0 and i % 100 == 0:
            print("top1 acc:    " + str(correct1/total) + "    top5 acc:   " + str(correct5/total) + "  time:   " + str(total_time/total))
    return correct1/total



# load data
valdir = os.path.join("val_subfolder")
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])),
    batch_size = 1,
    shuffle = False,
    num_workers = 4,
    pin_memory = True
)
print("Done: load data.")

# load pre-trained onnx model
session = onnxruntime.InferenceSession("resnet50v2/resnet50v2.onnx", None)
input_name = session.get_inputs()[0].name
print("Done: load onnx model. start to validate")
# labels = load_labels('imagenet-simple-labels.json')
acc1 = validate(val_loader, session)
print("acc1: " + str(acc1))