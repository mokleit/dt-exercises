import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class NoGPUAvailable(Exception):
    pass

class Wrapper():
    def __init__(self, model_file):
        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU
        # TODO If no GPU is available, raise the NoGPUAvailable exception


    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!

            box, label, score = self.model.predict(img) # TODO you probably need to send the image to a tensor, etc.
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores

class Model():    # TODO probably extend a TF or Pytorch class!
    def __init__(self):
        # TODO Instantiate your weights etc here!
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
 
        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 5  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    #   TODO add your own functions if need be!
