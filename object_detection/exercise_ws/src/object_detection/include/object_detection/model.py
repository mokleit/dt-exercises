import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch
import utils
import transforms as T
from dataset import Dataset
from torch import torch


# PATH = '/home/mokleit/dt-exercises/object_detection/sim/npz'
PATH = '/content/drive/My Drive/Fall2020/IFT6757/Exercise3/dt-exercises/object_detection/sim/npz'
MODEL_PATH = '/content/drive/My Drive/Fall2020/IFT6757/Exercise3/dt-exercises/object_detection/exercise_ws/src/object_detection/include/object_detection/weights/model_weights.txt'

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train():
    print('In train main')
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Device set')
    # our dataset has 5 classes
    num_classes = 5
    # use our dataset and defined transformations
    dataset = Dataset(PATH, get_transform(train=True))
    dataset_test = Dataset(PATH, get_transform(train=False))
    print('Datasets transformed')

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])
    print('Datasets split')

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
        
    print('Data set loader')

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    
    print('Model Loaded')

    # move model to the right device
    model.to(device)
    
    print('Model set to device')

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    
    print('Starting epochs')

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
        
    print("About to save!")
    torch.save(model.state_dict(), f"{MODEL_PATH}")
    print("That's it!")