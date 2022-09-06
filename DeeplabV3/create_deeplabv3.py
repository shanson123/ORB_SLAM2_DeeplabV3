import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

import cv2

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

class myModel(torch.nn.Module):
    def __init__(self, model):
        super(myModel, self).__init__()
        self.model = model
    def forward(self, x):
        x = self.model(x)["out"]
        return x


def main():
    aux = False  # inference time not need aux_classifier
    classes = 20
    # weights_path = "./deeplabv3_resnet50_coco.pth"
    img_path = "./test3.png"
    palette_path = "./palette.json"
    # assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    deeplabV3 = torch.hub.load('pytorch/vision:v0.12.0', 'deeplabv3_resnet50', pretrained=True)
    model = myModel(deeplabV3)
    # model = deeplabv3_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    # weights_dict = torch.load(weights_path, map_location='cpu')
    # for k in list(weights_dict.keys()):
    #     if "aux" in k:
    #         del weights_dict[k]

    # load weights
    # model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    # original_img = Image.open(img_path)
    img = cv2.imread(img_path)
    size = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # if size[1] > size[0]:
    #     ratio = 520.0 / size[0]
    # else:
    #     ratio = 520.0 / size[1]
    # img = cv2.resize(img, (0, 0), img, ratio, ratio, cv2.INTER_LINEAR)
    # img = cv2.resize(img, (520, 520), img, 0, 0, cv2.INTER_LINEAR)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([
        # transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))
        print(output.shape)

        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        mask.save("test_result.png")
        traced_model = torch.jit.trace(model, img.to(device))
        traced_model.save("DeeplabV3.pt")


if __name__ == '__main__':
    main()
