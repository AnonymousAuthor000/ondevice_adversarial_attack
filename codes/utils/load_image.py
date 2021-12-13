import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ImageNet1000(Dataset):
    def __init__(self, csv_path, file_path, transform=None, target=False):
        self.csv_path = csv_path
        self.file_path = file_path
        self.transform = transform
        self.target = target

        self.data_info = pd.read_csv(csv_path, header=None)
        self.img_id = np.asarray(self.data_info.iloc[1:, 0])
        self.img_label = np.asarray(self.data_info.iloc[1:, 6])
        self.img_tarlabel = np.asarray(self.data_info.iloc[1:, 7])
        # print(self.img_label[0], self.img_tarlabel[0])

        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        img_name = self.img_id[index]
        img = Image.open(self.file_path + img_name + '.png')
        if self.transform is not None:
            img = self.transform(img)
        if self.target is False:
            label = self.img_label[index]
        else:
            label = self.img_tarlabel[index]
        return img, int(label)-1

    def __len__(self):
        return self.data_len


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
# transforms = transforms.Compose([transforms.Resize(256),
#                                  transforms.CenterCrop(224),
#                                  transforms.ToTensor(),
#                                  normalize,])

# mydataset = ImageNet1000(csv_path='/datassd/Dataset/ImageNet1000/dev_dataset.csv',
#                          file_path='/datassd/Dataset/ImageNet1000/images',
#                          transform=transforms)
# testloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=False)

# print(mydataset.data_info)
