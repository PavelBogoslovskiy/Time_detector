import cv2
import torch
from func_module import crop_img, transform_img, load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load classification model
model_cls = load_model()
model_cls.to(device)
model_cls.eval()


def predictor(img_path):
    img = cv2.imread(img_path)
    # Detect clock + crop + some transformations
    img_c = crop_img(img)
    if img_c[1] == 0:
        return 'Clock not found :-(', 0
    img_c = cv2.resize(img_c[0], (224, 224)) / 255  # resize
    img_t = transform_img(img_c)  # to tensor + normalize
    img_t = img_t.to(torch.float32)  # to float32
    img_t = torch.unsqueeze(img_t, 0)  # expand dimension
    # Predict
    output = model_cls(img_t.to(device))
    _, prediction = torch.max(output.data, 1)
    prediction = prediction.item()
    hours = prediction // 60
    minutes = prediction % 60

    if len(str(hours)) == 1:
        hours = '0' + str(hours)
    if len(str(minutes)) == 1:
        minutes = '0' + str(minutes)
    time = f'{hours} : {minutes}'

    return time, 1
