import librosa as lb
import os
import time

"""
Point is to find out how many FSD50k files are longer than 10 seconds which is
a plausible clip length. If many files are longer than 10 then using 10 second 
threshold might not be a good idea.
"""
PATH_TO_FSD50k_DEV_DIR = r'S:\machine_listening\Datasets\FSD50K\FSD50K.dev_audio'

test_files = os.scandir(PATH_TO_FSD50k_DEV_DIR)
songs_over_10_secs = 0
counter = 0

file_number_limit = 100
start = time.perf_counter()
for entry in test_files:
    duration = lb.get_duration(path=entry.path)
    if duration > 10:
        songs_over_10_secs += 1
    counter += 1
    if counter == file_number_limit:
        break
print(f"{songs_over_10_secs} out of {counter} were over 10 seconds long")
end = time.perf_counter()
print(f"{file_number_limit} files took {end-start}")
    