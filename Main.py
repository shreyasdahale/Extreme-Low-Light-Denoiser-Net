import numpy as np
# import matplotlib.pyplot as plt
# import imageio
import cv2
import os
# import PIL

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets
# from torchvision import transforms
from tqdm import tqdm 
from PIL import Image


test = []
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        img = cv2.imread(filepath)
        if img is not None:
            images.append(img)
    return images

test = load_images_from_directory("./test/low/")


test.transpose(0, 3, 1, 2)
test = torch.tensor(test, dtype=torch.float32).cuda()


class CNNBlock(nn.Module):
    def __init__(self, filters):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(3, filters, kernel_size=3, padding=1)
        self.sp = nn.Softplus()

    def forward(self, x):
        out = self.conv(x)
        out = self.sp(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.sp1 = nn.Softplus()
        self.sp2 = nn.Softplus()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.sp1(out)
        out = self.conv2(out)
        out += residual  
        out = self.sp2(out)
        return out

class CAM(nn.Module):
    def __init__(self, channels):
        super(CAM, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.Softplus(),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.global_avg_pool(x)
        y = self.fc(y)
        return x * y

class PyramidPooling(nn.Module):
    def __init__(self, pool_size, output_size):
        super(PyramidPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.output_size = output_size

    def forward(self, x):
        pooled = self.pool(x)
        upsampled = F.interpolate(pooled, size=self.output_size, mode='bilinear', align_corners=True)
        return upsampled

class KSM(nn.Module):
    def __init__(self):
        super(KSM, self).__init__()
        self.conv = nn.Conv2d(36, 3, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block1 = CNNBlock(filters=32)
        self.residual_block = ResidualBlock(filters=32)
        self.cam = CAM(channels=32)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.pyramid1 = PyramidPooling(pool_size=1, output_size=(400, 600))
        self.pyramid2 = PyramidPooling(pool_size=2, output_size=(400, 600))
        self.pyramid3 = PyramidPooling(pool_size=4, output_size=(400, 600))
        self.pyramid4 = PyramidPooling(pool_size=8, output_size=(400, 600))
        self.pyramid5 = PyramidPooling(pool_size=16, output_size=(400, 600))
        self.ksm = KSM()
        self.final_conv = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.sp = nn.Softplus()

    def forward(self, x):
        C1 = self.conv_block1(x)
        C1 = self.residual_block(C1) 
        cam = self.cam(C1)
        C2 = self.sp(self.conv2(cam))
        concat1 = torch.cat([C2, x], dim=1)  

        p1 = self.pyramid1(concat1)
        p2 = self.pyramid2(concat1)
        p3 = self.pyramid3(concat1)
        p4 = self.pyramid4(concat1)
        p5 = self.pyramid5(concat1)

        concat2 = torch.cat([p1, p2, p3, p4, p5, concat1], dim=1) 

        ksm = self.ksm(concat2)
        out = self.final_conv(ksm)
        return out

    
model = Net().cuda()  

model.load_state_dict(torch.load('./nnweights3.pth'))
model.eval()



batch_size = 16
combined_dataset_test = TensorDataset(test)
testing_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


reconstructed_images = []

with torch.no_grad():
    for images in tqdm(testing_loader):
        images = images[0].cuda()
        outputs = model(images)
        outputs = outputs.cpu().numpy()
        outputs = outputs * 255
        outputs = outputs.astype(np.uint8)
        reconstructed_images.extend(outputs)



output_dir = './test/predicted/'
os.makedirs(output_dir, exist_ok=True)

for i, img_array in enumerate(reconstructed_images):
    img = Image.fromarray(img_array, mode='RGB')
    img.save(os.path.join(output_dir, f'image_{i}.png'))

