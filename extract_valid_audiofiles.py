import os
import zipfile as zf
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description="A script to extract Audioset files " \
"from a directory containing several blob files and extracting to destination directory based on an " \
"external text file of valid filenames. Archive's file structure is ignored in the process.")
parser.add_argument("--blob_dir", help="Path of the blob directory")
parser.add_argument("--valid_filenames", help="Text file containing valid filenames separated by '\n' ")
parser.add_argument("--target_dir", help="The destination of the extracted files")

args = parser.parse_args()

# Check that all arguments were provided, exit otherwise
if not (args.blob_dir and args.valid_filenames and args.target_dir):
    print("Please provide all the arguments")
    sys.exit()

PATH_TO_AS_TRAIN_FNAMES = r'C:\Users\mp431591\Documents\work_code\cl_30\continual_learning\audioset_train_filenames_top50.txt'
valid_filenames = set()
with open(args.valid_filenames, 'r') as fnames:
    for line in fnames.readlines():
        valid_filenames.add(line.rstrip('\n'))

# These same files are found on a computing cluster, testing locally first 
PATH_TO_LOCAL_AUDIOSET_00_ZIP = r'C:\Users\mp431591\Documents\audioset_unbalanced_test\unbalanced_train_segments_part00_full.zip'
PATH_TO_TEST_DEST = r'C:\Users\mp431591\Documents\audioset_unbalanced_test\test_dest'
PATH_TO_BLOB_DIR = r'C:\Users\mp431591\Documents\audioset_unbalanced_test\faux_blob_folder_for_testing'

with os.scandir(args.blob_dir) as blob_dir:
    for entry in blob_dir:

        with zf.ZipFile(entry.path, 'r') as archive:
            archive_path = archive.filename

            for zip_info in archive.infolist():

                if zip_info.is_dir():
                    continue
                
                zip_info.filename = os.path.basename(zip_info.filename)
                filename = zip_info.filename
                clean_file_name = filename.rstrip(".wav")

                if clean_file_name in valid_filenames:
                    archive.extract(zip_info, path=args.target_dir)
        print(f"Extracted valid files from dir: {entry.name}")


