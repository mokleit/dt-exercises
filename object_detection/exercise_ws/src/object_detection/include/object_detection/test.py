import numpy as np
import os
from PIL import Image
from torch import torch
from cv2 import cv2

PATH = '/home/mokleit/dt-exercises/object_detection/sim/npz'
# npz_files = os.listdir(PATH)

# images = []
# bboxes = []
# classes = []

# for i in range(100):
#     npz_file = np.load(PATH + '/' + str(i) + '.npz')
#     image = Image.fromarray(npz_file['arr_0'].astype('uint8')).convert('RGB')
#     images.append(image)
#     bboxes.append(npz_file['arr_1'])
#     classes.append(npz_file['arr_2'])

# idx = 50

# img = images[idx]
# # load boxes
# boxes = torch.as_tensor(bboxes[idx], dtype=torch.float32)
# # load labels
# labels = torch.as_tensor(classes[idx], dtype=torch.uint8)
# # define image id
# image_id = torch.tensor([idx])
# # compute area
# area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
# # suppose al instances are not crowd
# iscrowd = torch.zeros(len(labels), dtype=torch.int64)

# # define target dictionnary
# target = {}
# target["boxes"] = boxes
# target["labels"] = labels
# target["image_id"] = image_id
# target["area"] = area
# target["iscrowd"] = iscrowd


# ranges = {
#     'cone': {'lower': (167,100,94), 'upper': (255,119,110)},
#     'bus': {'lower': (189,119,0), 'upper': (255,255,30)},
#     'duckie': {'lower': (86,92,200), 'upper': (150,140,255)},
#     'truck': {'lower': (107,90,100), 'upper': (146,125,140)},
#     'background': {'lower': (224,0,246), 'upper': (255,10,255)}
# }

# kernels = {
#     'background': {'kernel': (15,15), 'lower': 100},
#     'bus': {'kernel': (7,7), 'lower': 120},
#     'cone': {'kernel': (5,5), 'lower': 115},
#     'duckie': {'kernel': (3,3), 'lower': 115},
#     'truck': {'kernel': (7,7), 'lower':105}
# }

name = 'bus'
num = '2111'

im1 = cv2.imread('/home/mokleit/dt-exercises/object_detection/sim/mask/' + name+ num+'.png')
im2 = cv2.imread('/home/mokleit/dt-exercises/object_detection/sim/raw/raw_image_'+num+'.png')
# # kernel = np.ones((6,6),np.uint8)
# # filtered = cv2.morphologyEx(im2, cv2.MORPH_OPEN, kernel)
# duckie_mask = cv2.inRange(im2, ranges[name]['lower'], ranges[name]['upper'])
# temp = im2.copy()
# temp[duckie_mask == 0] = (0,0,0)
# kernel = np.ones(kernels[name]['kernel'], np.uint8)
# filtered = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)

# # Convert to gray scale
# gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, kernels[name]['lower'], 255, 0)
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for countour in contours:
#     x,y,w,h = cv2.boundingRect(countour)
#     cv2.rectangle(filtered,(x,y),(x+w,y+h),(0,255,0),2) 
#     box_coordinates = [x, y, x + w, y + h]


cv2.imshow('Mask', im1)
cv2.imshow('Raw', im2)
cv2.waitKey(0)