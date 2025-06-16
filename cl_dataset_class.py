import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class CL_dataset(Dataset):
    def __init__(self, path_to_data_hdf5, dataset: str, split: str, nr_of_classes: int):

        self.path_to_data_hdf5 = path_to_data_hdf5
        self.hdf5_grp_name = ""
        self.dataset = dataset
        self.split = split
        self.nr_of_classes = nr_of_classes

        if dataset not in ['fsd50k', 'audioset']:
            raise ValueError("'dataset' must be either 'fsd50k' or 'audioset'.")
        
        if split not in ['train', 'eval']:
            raise ValueError("'split' must be either 'train' or 'eval'.")
        
        if nr_of_classes not in [30, 35, 40, 45, 50]:
            raise ValueError("'number_of_classes' should be in [30, 35, 40, 45, 50].")
        

        self.fnames = []
        self._collect_filenames()
        

    def __getitem__(self, index):
        fname = self.fnames[index]
        melspec, label = self._load_melspec_and_label(fname)
        return melspec, label
    
    def __len__(self):
        return len(self.fnames)
    
    def _load_melspec_and_label(self, fname):
        nr_of_classes = self.nr_of_classes
        label_val = 'label_' + str(nr_of_classes)
        melspec = 0
        label = 0

        with h5py.File(self.path_to_data_hdf5, 'r') as data:
            grp = data[self.hdf5_grp_name]

            # TODO: Unify the label and membership attributes for when 
            # nr of classes is 50
            if self.nr_of_classes == 50: 
                melspec = torch.from_numpy(grp[fname][0]).unsqueeze(0)
                label = grp[fname].attrs['label']
            else:
                melspec = torch.from_numpy(grp[fname][0]).unsqueeze(0)
                label = grp[fname].attrs[label_val]
        return melspec, label 

    
    def _collect_filenames(self):
        dataset = self.dataset
        split = self.split
        nr_of_classes = self.nr_of_classes

        with h5py.File(self.path_to_data_hdf5, 'r') as data:

            grp = ""

            for grp_name in data.keys():
                if (dataset in grp_name and
                    split in grp_name):
                    grp = data[grp_name]
                    self.hdf5_grp_name = grp_name

            for fname in grp:
                if self.nr_of_classes == 50:
                    self.fnames.append(fname)
                else:
                    membership_val = 'in_' + str(nr_of_classes)
                    if grp[fname].attrs[membership_val] == 1:
                        self.fnames.append(fname)

def main():
    PATH_TO_DATA_HDF5 = r'C:\Users\mp431591\Documents\work_code\cl_30\continual_learning\data_cl.hdf5'
    test_data = CL_dataset(path_to_data_hdf5=PATH_TO_DATA_HDF5,
                           dataset='fsd50k',
                           split='eval',
                           nr_of_classes=50)
    number_of_files = len(test_data)
    print(number_of_files)

    for i in range(5):
        random_idx = np.random.randint(0, number_of_files)
        rand_melspec, rand_label = test_data[random_idx]
        print(f"Random melspec: {rand_melspec.shape}")
        print(f"Corresponding label: {rand_label}")

main()


