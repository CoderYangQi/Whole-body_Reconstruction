import torch
import SimpleITK as sitk
import random
import numpy as np

class VolumeData(torch.utils.data.Dataset):
    def __init__(self, image_file, block_size=(64, 64, 64)):
        self.image_file = image_file
        self.block_size = block_size
        self.init = False
        img = sitk.ReadImage(self.image_file[0])
        self.size = [img.GetSize()[0] // self.block_size[0],
                     img.GetSize()[1] // self.block_size[1],
                     len(image_file) // self.block_size[2]]

    def _real_init(self):
        self.image = sitk.ReadImage(self.image_file)

    def __getitem__(self, item):
        if not self.init:
            self._real_init()
            self.init = True
        pos = (item % self.size[0],
               (item % (self.size[0] * self.size[1])) // self.size[0],
               item // (self.size[0] * self.size[1]))
        patch = sitk.GetArrayFromImage(self.image[pos[0] * self.block_size[0]: (pos[0] + 1) * self.block_size[0],
                                       pos[1] * self.block_size[1]: (pos[1] + 1) * self.block_size[1],
                                       pos[2] * self.block_size[2]: (pos[2] + 1) * self.block_size[2]])
        return np.float32([patch])

    def __len__(self):
        return self.size[0] * self.size[1] * self.size[2]