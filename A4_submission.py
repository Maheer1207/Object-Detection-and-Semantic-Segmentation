# To work on assignment 4, I have used the follwoing resources:
#   1. https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#   2. https://arxiv.org/abs/1505.04597
#   3. https://pytorch.org/docs/stable/index.html
#   4. https://pytorch.org/docs/stable/index.html
#   5. https://github.com/Leen-Alzebdeh/YOLOv5-UNet-Double-MNIST?tab=readme-ov-file

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.hub
import cv2


class UNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UNet, self).__init__()
    self.conv1 = self.contract_block(in_channels, 32, 7, 3)
    self.conv2 = self.contract_block(32, 64, 3, 1)
    self.conv3 = self.contract_block(64, 128, 3, 1)

    self.upconv3 = self.expand_block(128, 64, 3, 1)
    self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
    self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

  def forward(self, x):
    conv1 = self.conv1(x)
    conv2 = self.conv2(conv1)
    conv3 = self.conv3(conv2)

    upconv3 = self.upconv3(conv3)
    upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
    upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
    return upconv1

  def contract_block(self, in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )

  def expand_block(self, in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    )   


def segmentate(images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = images.shape[0]
    
    # Load the model
    model = UNet(3, 11).to(device)
    model.load_state_dict(torch.load("unet_model_with_nll_loss_25_1e-4.pth", map_location=device), strict=False)
    model.eval()

    # Reshape images to [N, 64, 64, 3] and normalize
    images = images.reshape(N, 64, 64, 3) / 255.0  # Reshape and normalize
    images = images.transpose(0, 3, 1, 2)  # Change to [N, 3, 64, 64] for PyTorch
    images = torch.tensor(images, dtype=torch.float32).to(device)

    # Process images in batch
    with torch.no_grad():
        outputs = model(images)  # Run the model on the batch
        pred_seg = outputs.argmax(dim=1).cpu().numpy().reshape(N, 4096)  # Convert to shape [N, 4096]

    return pred_seg


# Load the YOLOv5 model for object detection
def load_yolov5_model(weights_path, device):
  yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True, _verbose=False)
  return yolov5.to(device)


def preprocess_images(N, images):
  # Reshape vectorized image to 64x64x3 for compatibility with YOLO models
  return images.reshape(N, 64, 64, 3)


# Limit predictions to top two bounding boxes with highest confidence scores
def limit_predictions(yolov5_results):
  max_conf = [0, 0]
  select_preds = torch.zeros(2, 6)  # Container for two predictions
  for preds in yolov5_results[0]:
    if preds[4] > max_conf[0]:
      max_conf[0] = preds[4]
      select_preds[0] = preds
    elif preds[4] > max_conf[1]:
      max_conf[1] = preds[4]
      select_preds[1] = preds
  return select_preds


# Generate predicted bounding boxes and class labels
def get_predictions(pred_class, pred_bboxes, yolov5_results, indx):
  predictions = np.array([yolov5_results[0][5], yolov5_results[1][5]], dtype=np.int32)
  indices = np.argsort(predictions)
  pred_class[indx] = predictions[indices]
  pred_box = get_bboxes_prediction(yolov5_results)
  pred_box = pred_box[indices]
  pred_bboxes[indx] = pred_box


def get_bboxes_prediction(yolov5_results):
  bbox_prediction = np.empty((2, 4), dtype=np.int32)
  bbox_prediction[0] = np.array(
    [yolov5_results[0][0], yolov5_results[0][1], yolov5_results[0][2], yolov5_results[0][3]], dtype=np.int32)
  bbox_prediction[1] = np.array(
    [yolov5_results[1][0], yolov5_results[1][1], yolov5_results[1][2], yolov5_results[1][3]], dtype=np.int32)
      
  return bbox_prediction


def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.zeros((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.zeros((N, 4096), dtype=np.int32)

    yolov5_weights_path = 'best_64_15.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pred_seg = segmentate(images)
    
    images = preprocess_images(N, images)

    yolov5_model = load_yolov5_model(yolov5_weights_path, device)

    for i in range(N):
        yolov5_results = yolov5_model(images[i], 64)
        yolov5_results = limit_predictions(yolov5_results.pred)
        get_predictions(pred_class, pred_bboxes, yolov5_results, i)

    return pred_class, pred_bboxes, pred_seg