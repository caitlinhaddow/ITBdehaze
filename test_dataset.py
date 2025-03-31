from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os

##### CH add
from utils_test import cropping
class dehaze_test_dataset(Dataset):
    def __init__(self, test_dir, crop_method):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test=[]
        for line in open(os.path.join(test_dir, 'test.txt')):
            line = line.strip('\n')
            if line!='':
                self.list_test.append(line)

        self.root_hazy = os.path.join(test_dir , 'hazy/')
        self.root_clean = os.path.join(test_dir , 'clean/')
        self.file_len = len(self.list_test)
        self.crop_method = crop_method
        # print(self.list_test)

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy + self.list_test[index])
        clean = Image.open(self.root_clean + self.list_test[index])
        hazy = self.transform(hazy)
        clean = self.transform(clean)

        if hazy.shape[0] == 4:
            assert torch.equal(hazy[-1:, :, :], torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = hazy[:3, :, :]
        if clean.shape[0] == 4:
            assert torch.equal(clean[-1:, :, :], torch.ones(1, clean.shape[1], clean.shape[2])), "hazy[-1:, :, :] is not all ones"
            clean = clean[:3, :, :]

        hazy, vertical = cropping(hazy, self.crop_method)

        return hazy, vertical, clean

    def __len__(self):
        return self.file_len
#####################


# class dehaze_test_dataset(Dataset):
#     def __init__(self, test_dir):
#         self.transform = transforms.Compose([transforms.ToTensor()])

#         self.list_test=[]
#         for line in open(os.path.join(test_dir, 'test.txt')):
#             line = line.strip('\n')
#             if line!='':
#                 self.list_test.append(line)

#         self.root_hazy = os.path.join(test_dir , 'hazy/')
#         self.root_clean = os.path.join(test_dir , 'clean/')
#         self.file_len = len(self.list_test)

#     def __getitem__(self, index, is_train=True):
#         hazy = Image.open(self.root_hazy + self.list_test[index])
#         clean = Image.open(self.root_clean + self.list_test[index])
#         hazy = self.transform(hazy)
#         # print(hazy.shape)

#         ### CH add
#         # width, height = hazy.size  # (6000, 4000)
#         # crop_size = min(width, height)  # 4000
#         # left = (width - crop_size) // 2
#         # top = (height - crop_size) // 2
#         # right = left + crop_size
#         # bottom = top + crop_size

#         # hazy = hazy.crop((left, top, right, bottom))  # Center crop to 4000x4000
#         # hazy = self.transform(hazy)
#         # if hazy.shape[0] == 4:
#         #     assert torch.equal(hazy[-1:, :, :], torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
#         #     hazy = hazy[:3, :, :]
        
#         # clean = clean.crop((left, top, right, bottom))
#         ####

#         hazy_up = hazy[:,0:640,:]
#         hazy_down = hazy[:,560:1200,:]

        

#         clean = self.transform(clean)

#         return hazy_up, hazy_down, hazy, clean

#     def __len__(self):
#         return self.file_len





