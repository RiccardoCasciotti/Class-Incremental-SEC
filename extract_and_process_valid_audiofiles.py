import os
import zipfile as zf
import argparse
import sys

import numpy as np
import h5py
import librosa as lb
import torchaudio
import torch

import utils.config as config
import utils.utility_funcs as my_utils

parser = argparse.ArgumentParser(description="A script to extract Audioset files " \
"from a directory containing several blob files and extracting to destination directory based on an " \
"external text file of valid filenames. Archive's file structure is ignored in the process.")
parser.add_argument("--blob_dir", help="Path of the blob directory")
parser.add_argument("--valid_filenames", help="Text file containing valid filenames separated by '\n' ")
parser.add_argument("--hdf5_filepath", help="The path where the HDF5 file is going to be")
parser.add_argument("--hdf5_group", help="the group name which the datasets will be under")
parser.add_argument("--pckl_label_dict", help="A python dict that contains filename : multihot label")

args = parser.parse_args()

# Check that all arguments were provided, exit otherwise
if not (args.blob_dir and args.valid_filenames
            and args.hdf5_filepath and args.hdf5_group
            and args.pckl_label_dict):
    print("Please provide all the arguments")
    sys.exit()

valid_filenames = set()
with open(args.valid_filenames, 'r') as fnames:
    for line in fnames.readlines():
        valid_filenames.add(line.rstrip('\n'))

hdf5_filename = args.hdf5_filepath
group_name = args.hdf5_group

# Creating the HDF5 file, and the group
with (h5py.File(hdf5_filename, "w")) as f:
    group = f.create_group(group_name)

# Get the pickled dict that connects the filename to its multihot label
PATH_TO_PCKL_LABEL_DICT = args.pckl_label_dict
audioset_train_labels = my_utils.pickle_load(PATH_TO_PCKL_LABEL_DICT)

# Create the melspec transformation
MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate,
                                                     n_fft=config.n_fft,
                                                     hop_length=config.hop_length,
                                                     n_mels=config.n_mels,
                                                     f_min=config.fmin,
                                                     f_max=config.fmax)

# Open the hdf5 file, remember to close
hdf5 = h5py.File(hdf5_filename, 'w')

# Take note of files that failed to open
bad_files = open("bad_filenames.txt", 'w')

compared_files = 0

# Open dir full of blobs/zip files
with os.scandir(args.blob_dir) as blob_dir:
    
    # For each blob
    for idx, entry in enumerate(blob_dir):
        print(f"Going through blob nr: {idx+1}", flush=True)
        print(f"Compared {compared_files} files so far")

        if (idx+1) != 7: # Testing jumping over faulty files
            continue

        with zf.ZipFile(entry.path, 'r') as archive:
            archive_path = archive.filename

            for zip_info in archive.infolist():

                compared_files += 1

                if zip_info.is_dir():
                    continue
                # Break archive directory structures
                zip_info.filename = os.path.basename(zip_info.filename) 
                filename = zip_info.filename
                clean_file_name = filename.rstrip(".wav")

                if clean_file_name in valid_filenames:
                    
                    valid_filenames.remove(clean_file_name)

                    # If extracting
                    #archive.extract(zip_info, path=PATH_TO_TEST_DEST)

                    # If processing
                    with archive.open(zip_info, 'r') as audio_file:
                        
                        try:
                            audio, sr = torchaudio.load(audio_file)
                            target_sr = config.sample_rate

                            if sr != target_sr:
                                print(f"Resampling {clean_file_name}")
                                audio = torchaudio.functional.resample(audio, 
                                                            orig_freq=sr,
                                                            new_freq=target_sr)
                            audio = my_utils.pad_or_truncate(audio, target_sr * 10)
                            mel_specgram = MEL_TRANSFORM(audio)
                            
                            mel_specgram = torch.transpose(mel_specgram, 2, 1)
                            mel_specgram = torch.log(mel_specgram + torch.finfo(torch.float32).eps)

                            hdf_path = group_name + '/' + clean_file_name
                            hdf5[hdf_path] = mel_specgram.numpy()
                            # Meta table filenames are without Ys and filenames are with Ys
                            clean_file_name_wo_Y = clean_file_name[1::]
                            hdf5[hdf_path].attrs['label'] = audioset_train_labels[clean_file_name_wo_Y]
                        except Exception as ex:
                            print(f"file {clean_file_name} from archive {archive} caused \
                                  an exception {ex}. Trying the next file", flush=True)
                            bad_files.write(clean_file_name + '\n')
                            continue
                            

hdf5.close()
bad_files.close()

print(f"Compared {compared_files} files", flush=True)
print("Finished processing files")