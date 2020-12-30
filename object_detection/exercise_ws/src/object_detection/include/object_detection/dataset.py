import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


PATH = '/home/mokleit/dt-exercises/object_detection/sim/npz'

class Dataset(object):
    def __init__(self, root=PATH):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        npz_files = os.listdir(root)
        self.images = []
        self.bboxes = []
        self.classes = []
        for i in range(len(npz_files)):
            npz_file = np.load(PATH + '/' + str(i) + '.npz')
            image = Image.fromarray(npz_file['arr_0'].astype('uint8')).convert('RGB')
            self.images.append(image)
            self.bboxes.append(npz_file['arr_1'])
            self.classes.append(npz_file['arr_2'])

    def __getitem__(self, idx):
        # load image
        img = self.images[idx]
        img = transforms.ToTensor()(img)
        # load boxes
        boxes = torch.as_tensor(self.bboxes[idx], dtype=torch.float32)
        # load labels
        labels = torch.as_tensor(self.classes[idx], dtype=torch.uint8)
        # define image id
        image_id = torch.tensor([idx])
        # compute area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose al instances are not crowd
        iscrowd = torch.zeros(len(labels), dtype=torch.int64)

        # define target dictionnary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.images)