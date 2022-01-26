import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_det = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model_det.eval()


# Detection function
def get_prediction(img, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).to(device)
    pred = model_det([img])
    pred_classes = pred[0]['labels'].cpu().numpy()
    pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
    pred_score = pred[0]['scores'].detach().cpu().numpy()
    flag = -1
    for i in range(len(pred_classes)):
        if pred_classes[i] == 85:
            flag = i
            break
    if flag == -1 or pred_score[flag] < threshold:
        flag = 0
        return None, flag
    else:
        x1, y1, x2, y2 = pred_boxes[flag]
        pred_box = [(x1, y1), (x2, y2)]
        flag = 1
        return pred_box, flag


# Crop function
def crop_img(img):
    pred = get_prediction(img)
    if pred[1] == 0:
        return None, 0
    (x1, y1), (x2, y2) = np.array(pred[0], dtype=int)
    crop_img = img[y1:y2, x1:x2]
    return crop_img, 1


def transform_img(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_t = transform(img)
    return img_t


# Load classification model
def load_model():
    # Load and finetuning pretrained model
    model_cls = models.resnext50_32x4d(pretrained=True)
    for param in model_cls.parameters():
        param.require = False
    num_ftrs = model_cls.fc.in_features
    model_cls.fc = nn.Linear(num_ftrs, 720)

    # Load saved weigths
    param_file_name = 'model_cls.pth'
    model_param = os.path.join(param_file_name)
    model_cls.load_state_dict(torch.load(model_param))
    print(model_cls)
    return model_cls
