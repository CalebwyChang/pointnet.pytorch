from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from utils import helpers
from path import Path
from torchvision import transforms

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

def default_transforms(npoints=1024):
    return transforms.Compose([
                                helpers.PointSampler(npoints),
                                helpers.Normalize(),
                                helpers.ToTensor()
                              ])

class ModelNetDataset(data.Dataset):
    def __init__(self, root_dir, valid=False, folder="train", npoints=1024, transform=default_transforms()):
        self.root_dir = Path(root_dir)
        folders = [dir for dir in sorted(os.listdir(self.root_dir)) if os.path.isdir(self.root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms(npoints)
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = self.root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = helpers.read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    gen_modelnet_id(datapath)
    d = ModelNetDataset(root=datapath)
    # print(len(d))
    # print(d[0])