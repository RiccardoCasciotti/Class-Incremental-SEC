import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from utils import config

class CL_dataset(Dataset):
    def __init__(self, 
                 path_to_data_hdf5, 
                 dataset: str, 
                 split: str, 
                 nr_of_classes: int, 
                 cil_classes: int=0):

        self.path_to_data_hdf5 = path_to_data_hdf5
        self.dataset = dataset
        self.split = split
        self.nr_of_classes = nr_of_classes
        self.cil_classes = cil_classes
        self.cil = False

        if self.cil_classes != 0:
            self.cil = True
        self.hdf5_grp_name = ""
        self.pos_weight = 0

        if dataset not in ['fsd50k', 'audioset']:
            raise ValueError("'dataset' must be either 'fsd50k' or 'audioset'.")
        
        if split not in ['train', 'eval']:
            raise ValueError("'split' must be either 'train' or 'eval'.")
        
        if nr_of_classes not in [30, 35, 40, 45, 50]:
            raise ValueError("'number_of_classes' should be in [30, 35, 40, 45, 50].")

        self.fnames = []
        if self.cil:
            self._collect_cil_filenames()
            print(f"Collecting cil filenames.")
        else:
            self._collect_filenames()
            print(f"Collecting files normally.")

        if split == 'train':
            self._get_pos_weight_val()
        

    def __getitem__(self, index):
        fname = self.fnames[index]
        melspec, label, fname = self._load_melspec_and_label(fname)
        return melspec, label, fname
    
    def __len__(self):
        return len(self.fnames)
    
    # The value used to compensate for the class imbalance
    # One possible approach used in CIL-ML-AUDIO
    def _get_pos_weight_val(self):
        nr_of_classes = self.nr_of_classes
        label_val = 'label_' + str(nr_of_classes)

        zeros = 0
        ones = 0

        with h5py.File(self.path_to_data_hdf5, 'r') as data:
            grp = data[self.hdf5_grp_name]

            for fname in grp:
                
                membership_val = 'in_' + str(nr_of_classes)
                if grp[fname].attrs[membership_val] == 1:
                    label = grp[fname].attrs[label_val]
                    one_count = np.count_nonzero(label)
                    zero_count = nr_of_classes - one_count 
                    ones += one_count
                    zeros += zero_count
        self.pos_weight = zeros / ones

    
    def _load_melspec_and_label(self, fname):
        nr_of_classes = self.nr_of_classes
        if self.cil:
            nr_of_classes += self.cil_classes
        label_val = 'label_' + str(nr_of_classes)
        melspec = 0
        label = 0

        with h5py.File(self.path_to_data_hdf5, 'r') as data:
            grp = data[self.hdf5_grp_name]

            melspec = torch.from_numpy(grp[fname][0]).unsqueeze(0)
            label = torch.from_numpy(grp[fname].attrs[label_val])
            label = label.type(torch.float32)
        return melspec, label, fname

    
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
                
                membership_val = 'in_' + str(nr_of_classes)
                if grp[fname].attrs[membership_val] == 1:
                    self.fnames.append(fname)
    
    def _collect_cil_filenames(self):
        dataset = self.dataset
        split = self.split
        nr_of_classes = self.nr_of_classes
        cil_classes = self.cil_classes


        with h5py.File(self.path_to_data_hdf5, 'r') as data:

            grp = ""

            for grp_name in data.keys():
                if (dataset in grp_name and
                    split in grp_name):
                    grp = data[grp_name]
                    self.hdf5_grp_name = grp_name

            for fname in grp:
                
                cil_mmbrshp_val = f"in_{str(nr_of_classes + cil_classes)}"
                cil_label_val = f"label_{nr_of_classes + cil_classes}"

                # Exclusive logic for getting only unseen files, results are not promising
                """membership_val = 'in_' + str(nr_of_classes)
                if grp[fname].attrs[membership_val] == 0:
                    if grp[fname].attrs[cil_mmbrshp_val] == 1:
                        self.fnames.append(fname)"""
                # More inclusive logic: files that have been seen before can be used as well
                if grp[fname].attrs[cil_mmbrshp_val] == 1:
                    cil_label = grp[fname].attrs[cil_label_val]

                    # Check if there is an appearance of a cil label
                    if np.any(cil_label[-cil_classes:]):
                        self.fnames.append(fname)
    
    def get_pos_weight(self):
        return torch.tensor(self.pos_weight)

# For quick testing
def main():
    PATH_TO_DATA_HDF5 = r'C:\Users\mp431591\Documents\work_code\cl_30\continual_learning\data_cl_complete.hdf5'
    test_data = CL_dataset(path_to_data_hdf5=PATH_TO_DATA_HDF5,
                           dataset='audioset',
                           split='train',
                           nr_of_classes=30, 
                           cil_classes=5)
    number_of_files = len(test_data)
    print(f"Number of files: {number_of_files}")

    for i in range(1):
        random_idx = np.random.randint(0, number_of_files)
        rand_melspec, rand_label, fname = test_data[random_idx]
        melspec_as_np = rand_melspec.squeeze(0).numpy().T
        print(f"Random melspec: {rand_melspec.shape}")
        print(f"Corresponding label: {rand_label.shape}")
        print(fname)
        print(rand_label)
        print(melspec_as_np.shape)
        print(f"melspec mean, std and var: {np.mean(melspec_as_np)}  {np.std(melspec_as_np)} {np.var(melspec_as_np)}")

        """
        fig, ax = plt.subplots()

        #S_dB = lb.power_to_db(melspec_as_np, ref=np.max)
        img = lb.display.specshow(melspec_as_np, x_axis='time',
                                y_axis='mel', sr=config.sample_rate,
                                fmax=config.fmax, ax=ax,
                                hop_length=config.hop_length)
        ax.set(title='Log mel spectrogram')
        plt.show()
        """

if __name__ == '__main__':
    main()

