import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


# PATH = '/home/mokleit/dt-exercises/object_detection/sim/npz'
PATH = '/content/drive/My Drive/Fall2020/IFT6757/Exercise3/dt-exercises/object_detection/sim/npz'

class Dataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        npz_files = os.listdir(self.root)
        self.images = []
        self.bboxes = []
        self.classes = []
        print("START")
        for i in range(len(npz_files)):
            print('NPZ', i)
            npz_file = np.load(PATH + '/' + str(i) + '.npz')
            image = Image.fromarray(npz_file['arr_0'].astype('uint8')).convert('RGB')
            self.images.append(image)
            self.bboxes.append(npz_file['arr_1'])
            self.classes.append(npz_file['arr_2'])

    def __getitem__(self, idx):
        # load image
        img = self.images[idx]
        # img = transforms.ToTensor()(img) 
        # load boxes
        boxes = torch.as_tensor(self.bboxes[idx], dtype=torch.float32)
        # load labels
        labels = torch.as_tensor(self.classes[idx], dtype=torch.int64)
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)