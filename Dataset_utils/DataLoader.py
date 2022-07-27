import os
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
#import sys
#sys.path.append("../Dataset_utils")
from .poses_parser import pose_2_sixd_array
#from poses_parser import pose_2_sixd_array
from torchvision import transforms



class UAV_GPS_Dataset(Dataset):
    def __init__(self, data_dir, representations=["rgb"],transform=None):
        self.data_dir = data_dir
        self.representations = representations
        self.images_names=os.listdir(os.path.join(data_dir,"rgb"))
        self.transform=transform
        self.TT=transforms.ToTensor()


    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image_name=self.images_names[idx]
        name=image_name.split(".")[0]
        sample={}
        for representation in self.representations:
            if representation=="rgb":
                repre=io.imread(os.path.join(self.data_dir,"rgb",(name+".png")))
                if repre.shape[2]>3:
                    repre=repre[:,:,:3]
                repre=self.TT(repre)
            elif representation=="poses":
                repre = np.loadtxt(
                    os.path.join(self.data_dir,"poses",(name+".txt")),
                    dtype=np.float16
                )
                repre=torch.Tensor(list(pose_2_sixd_array(repre)))
            elif representation=="semantics":
                repre=torch.from_numpy(
                    np.fromfile(
                    os.path.join(self.data_dir,"addl_scene_info","semantics",(name+".npy")),
                    dtype=np.uint8,
                    offset=128
                 ).reshape(480,720))
            else:
                repre = torch.load(os.path.join(self.data_dir,"addl_scene_info",representation,(name+".dat")))
                
            sample[representation]=repre
        if self.transform:
            sample=self.transform(sample)
        return sample